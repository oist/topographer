"""ScalarNetwork binding a singular scalar to a concrete Graph."""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
from typing import Any, Self

import rustworkx as rx

from topographer.graph import _archive
from topographer.graph._core import Graph
from topographer.graph._payloads import copy_graph_with_payloads


class ScalarNetwork:
    """A connected undirected Graph with declared singular scalar attributes.

    Examples
    --------
    >>> from topographer.graph import ScalarNetwork
    >>> issubclass(ScalarNetwork, object)
    True
    """

    def __init__(
        self,
        graph: Graph,
        *,
        scalar_attr: str,
        edge_scalar_attr: str | None = None,
    ) -> None:
        if not isinstance(graph, Graph):
            raise TypeError(f"Expected a Topographer Graph instance, got {type(graph).__name__}.")

        if graph.vertex_count == 0:
            raise ValueError("Graph cannot be empty.")

        if graph.directed:
            raise ValueError("ScalarNetwork requires an undirected graph.")

        # Check connectivity
        comps = graph.connected_components()
        if len(comps) > 1:
            raise ValueError(
                f"ScalarNetwork requires a connected graph, but found {len(comps)} disconnected components. "
                f"Component sizes: {[len(c) for c in comps]}"
            )

        if not isinstance(scalar_attr, str) or not scalar_attr:
            raise ValueError("scalar_attr must be a non-empty string.")

        if scalar_attr not in graph.vertex_attr_names:
            if scalar_attr == graph.vertex_id_attr:
                raise ValueError(
                    f"Vertex identity attribute {scalar_attr!r} cannot be selected as a scalar attribute."
                )
            raise ValueError(
                f"Scalar attribute {scalar_attr!r} not found among vertex attributes: {graph.vertex_attr_names}"
            )

        if edge_scalar_attr is not None:
            if not isinstance(edge_scalar_attr, str) or not edge_scalar_attr:
                raise ValueError("edge_scalar_attr must be a non-empty string.")
            if edge_scalar_attr not in graph.edge_attr_names:
                if edge_scalar_attr == graph.edge_id_attr:
                    raise ValueError(
                        f"Edge identity attribute {edge_scalar_attr!r} cannot be selected as an edge scalar attribute."
                    )
                raise ValueError(
                    f"Edge scalar attribute {edge_scalar_attr!r} not found among edge attributes: {graph.edge_attr_names}"
                )

        # Snapshot the internal graph
        self._graph_snapshot: rx.PyGraph = graph.graph  # type: ignore[assignment]
        self._vertex_ids = graph.vertex_ids
        self._scalar_attr = scalar_attr
        self._edge_scalar_attr = edge_scalar_attr
        self._vertex_id_attr = graph.vertex_id_attr
        self._edge_id_attr = graph.edge_id_attr

        # Validate vertex scalar values directly via vertex_id mapping
        self._vertex_scalars: dict[Any, float] = {}
        for vid in self._vertex_ids:
            old_idx = graph._vertex_id_to_index[vid]
            payload = graph._graph.get_node_data(old_idx)
            val = payload[scalar_attr]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"Scalar value on vertex must be real numeric, got {type(val).__name__}."
                )
            if not math.isfinite(val):
                raise ValueError(f"Scalar value {val} on vertex is not finite.")
            self._vertex_scalars[vid] = float(val)

        # Validate optional edge scalar
        if edge_scalar_attr is not None:
            for e_idx in graph._graph.edge_indices():
                payload = graph._graph.get_edge_data_by_index(e_idx)
                eval_val = payload[edge_scalar_attr]
                if isinstance(eval_val, bool) or not isinstance(eval_val, (int, float)):
                    raise TypeError(
                        f"Edge scalar value must be real numeric, got {type(eval_val).__name__}."
                    )
                if not math.isfinite(eval_val):
                    raise ValueError(f"Edge scalar value {eval_val} is not finite.")

    @property
    def scalar_attr(self) -> str:
        return self._scalar_attr

    @property
    def edge_scalar_attr(self) -> str | None:
        return self._edge_scalar_attr

    @property
    def vertex_id_attr(self) -> str:
        """Name of the vertex identity attribute of the source graph."""
        return self._vertex_id_attr

    @property
    def edge_id_attr(self) -> str | None:
        """Name of the declared edge identity attribute, or None when edge identities are generated."""
        return self._edge_id_attr

    @property
    def vertex_count(self) -> int:
        return len(self._vertex_ids)

    @property
    def edge_count(self) -> int:
        return self._graph_snapshot.num_edges()

    @property
    def vertex_ids(self) -> list[Any]:
        return list(self._vertex_ids)

    @property
    def graph(self) -> rx.PyGraph:
        """Return defensive copy of internal Rustworkx snapshot.

        The copy has its own node and edge payloads, so mutating it never alters this network.
        """
        return copy_graph_with_payloads(self._graph_snapshot)

    def write(self, path: str | os.PathLike[str]) -> None:
        """Write this network to a new folder; raise ``FileExistsError`` if the path exists.

        The folder holds ``vertices.parquet`` (one row per vertex in canonical order),
        ``edges.parquet`` (one row per edge in edge order, endpoints as native vertex
        identities) and ``manifest.json``, which is written last. Everything is validated
        before the folder is created.

        Raises
        ------
        TypeError
            If vertex or declared edge identities are not finite ``str``, ``int``, ``float``
            or ``bool`` of one exact type, if an attribute is not one of those types, or if an
            attribute is named ``source_id`` or ``target_id``.
        FileExistsError
            If ``path`` already exists.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from topographer.graph import Graph, ScalarNetwork
        >>> graph = Graph.from_tables(
        ...     pl.DataFrame({"vid": ["a", "b"], "value": [0.1, 0.9]}),
        ...     pl.DataFrame({"src": ["a"], "dst": ["b"]}),
        ...     vertex_id_col="vid", source_id_col="src", target_id_col="dst",
        ...     vertex_attribute_cols=["value"],
        ... )
        >>> network = ScalarNetwork(graph, scalar_attr="value")
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     network.write(Path(tmp) / "network")
        ...     sorted(p.name for p in (Path(tmp) / "network").iterdir())
        ['edges.parquet', 'manifest.json', 'vertices.parquet']
        """
        tables = _archive.graph_to_tables(
            self._graph_snapshot,
            vertex_ids=self._vertex_ids,
            vertex_id_attr=self._vertex_id_attr,
            edge_id_attr=self._edge_id_attr,
            edge_scalar_attr=self._edge_scalar_attr,
        )
        fields = {
            **tables.manifest_fields(),
            "network_class": type(self).__name__,
            "scalar_attr": self._scalar_attr,
            "edge_scalar_attr": self._edge_scalar_attr,
            **self._manifest_extras(),
        }
        folder = _archive.create_folder(path)
        try:
            tables.vertices.write_parquet(folder / _archive.VERTICES_TABLE)
            tables.edges.write_parquet(folder / _archive.EDGES_TABLE)
            _archive.write_manifest(folder, _archive.SCALAR_NETWORK_FORMAT, fields)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> Self:
        """Read a network folder written by `write`; returns an instance of ``cls``.

        Reading rebuilds the graph through ``Graph.from_tables`` and the normal constructor,
        so connectedness, finite scalars and unique identities are validated again.
        ``ScalarNetwork.read`` on a folder written from a subclass returns a plain
        ``ScalarNetwork`` and ignores the subclass extras.

        Raises
        ------
        ValueError
            If the folder, manifest or a table is missing, the format or format version is not
            supported, columns are missing, counts disagree with the manifest, or validation fails.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from topographer.graph import Graph, ScalarNetwork
        >>> graph = Graph.from_tables(
        ...     pl.DataFrame({"vid": ["a", "b"], "value": [0.1, 0.9]}),
        ...     pl.DataFrame({"src": ["a"], "dst": ["b"]}),
        ...     vertex_id_col="vid", source_id_col="src", target_id_col="dst",
        ...     vertex_attribute_cols=["value"],
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ScalarNetwork(graph, scalar_attr="value").write(Path(tmp) / "network")
        ...     ScalarNetwork.read(Path(tmp) / "network").vertex_ids
        ['a', 'b']
        """
        folder = Path(path)
        manifest = _archive.read_manifest(folder, _archive.SCALAR_NETWORK_FORMAT)
        graph = _archive.read_network_graph(folder, manifest)
        return cls._from_graph(graph, manifest)

    def _manifest_extras(self) -> dict[str, Any]:
        """Extra manifest entries a subclass stores; the base class has none."""
        return {}

    @classmethod
    def _from_graph(cls, graph: Graph, manifest: dict[str, Any]) -> Self:
        """Build an instance from the rebuilt graph and the manifest; subclasses override."""
        return cls(
            graph,
            scalar_attr=manifest["scalar_attr"],
            edge_scalar_attr=manifest["edge_scalar_attr"],
        )
