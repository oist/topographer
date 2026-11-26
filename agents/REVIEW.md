# REVIEW.md

Pull-request review checklist.

## Correctness and scope

- Does the change operate on a generic scalar network rather than assume biological sequences?
- Are vertex and edge scalar attributes named and validated at the public boundary?
- Is tie handling and output ordering deterministic?
- Are numerical assumptions explicit, including finite values and scalar direction?

## API and maintainability

- Is the public API small, documented, and importable from its intended module?
- Are compatibility paths preserved or their migration documented?
- Can a contributor understand each new function without following a deep class hierarchy?
- Are optional dependencies lazily imported where practical?

## Tests and artifacts

- Are new behaviors covered by small, deterministic tests?
- Does the cube fixture or another minimal graph demonstrate the expected topology?
- Do `uv run ruff format --check .`, `uv run ruff check .`, `uv run pytest -q`, and `uv build` pass?
- For plots, are configuration defaults and the emitted data/specification tested rather than a fragile image snapshot?

## Documentation and release hygiene

- Are README, API docs, and `CHANGELOG.md` updated when public behavior changes?
- Is the contribution/review guidance still accurate?
- Are examples located under generic examples or `topographer.biology`, not in the topology core?
