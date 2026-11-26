"""Check local links and fragments in the rendered documentation site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if value is None:
                continue
            if key == "id" or (tag == "a" and key == "name"):
                self.ids.add(value)
            if key in {"href", "src"}:
                self.links.append(value)


def check(site: Path, base_path: str, site_host: str = "oist.github.io") -> list[str]:
    site = site.resolve()
    pages: dict[Path, Page] = {}
    for path in site.rglob("*.html"):
        page = Page()
        page.feed(path.read_text(encoding="utf-8", errors="replace"))
        pages[path] = page

    if not pages:
        return [f"No HTML pages found in {site}"]

    errors = []
    prefix = "/" + base_path.strip("/") if base_path.strip("/") else ""
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.netloc and url.netloc != site_host:
                continue
            if url.scheme and not url.netloc:
                continue
            target_path = unquote(url.path)
            if target_path.startswith("/"):
                if prefix and (target_path == prefix or target_path.startswith(prefix + "/")):
                    target_path = target_path[len(prefix) :]
                target = site / target_path.lstrip("/")
            else:
                target = path.parent / target_path if target_path else path
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(site) or not target.exists():
                errors.append(f"{path.relative_to(site)}: missing target {link}")
            # great-docs currently emits links to API anchors that it does not
            # render. Check the target page but defer those generated anchors.
            elif (
                url.fragment
                and target in pages
                and not target.is_relative_to(site / "reference")
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(site)}: missing fragment {link}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--base-path", default="/topographer/")
    args = parser.parse_args()
    if not args.site.is_dir():
        parser.error(f"site directory does not exist: {args.site}")
    errors = check(args.site, args.base_path)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked local documentation links: {len(errors)} errors")
    print("Generated API reference fragments are excluded pending a great-docs fix.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
