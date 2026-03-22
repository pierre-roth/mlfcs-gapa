# Progress Report

## How To Use

Add a new section for each calendar week.

Within each week:

- keep the `Weekly Snapshot` short and project-level
- add one `Contributor Update` subsection per person
- append new weeks at the top so the latest status is easiest to find
- prefer links to commits, PRs, files, notebooks, or reports where relevant

Recommended conventions:

- Use ISO dates: `YYYY-MM-DD`
- Use one name consistently across weeks
- Keep bullets concrete and outcome-focused
- If something is blocked, say what is needed to unblock it

---

## Weekly Template

Copy this block for each new week.

```md
## Week of YYYY-MM-DD

### Weekly Snapshot

- Overall status:
- Main goal for the week:
- Biggest win:
- Biggest risk or blocker:

### Contributor Update: Name

- Focus area:
- Completed:
- In progress:
- Blocked:
- Next week:
- Links:

### Contributor Update: Name

- Focus area:
- Completed:
- In progress:
- Blocked:
- Next week:
- Links:
```

---

## Week of 2026-03-16

### Weekly Snapshot

- Overall status: Data preprocessing and validation workflow is now in place for Databento-based experiments.
- Main goal for the week: Convert, validate, and package usable project data so method work can proceed on top of a stable input pipeline.
- Biggest win: The converter was validated against `MBP-10`, a major reconstruction bug was fixed, and a tracked sample dataset was added for contributors without the full raw data.
- Biggest risk or blocker: The legacy training stack still needs a fully resolved `uv` dependency story, especially around `tensorflow` and `tensorforce`.
- Added complete ground-up implemetation of the paper's continuous part in `lobmm/`.

### Contributor Update: Pierre

- Focus area: Data pipeline, project setup, and reproducibility.
- Completed:
  - Added Databento conversion, validation, and divergence-analysis tooling.
  - Converted and validated `GOOGL` and `AAPL`.
  - Added a tracked sample dataset in `data/sample`.
  - Moved the repo toward a `uv`-first setup and removed the old conda environment file.
- In progress:
  - Finalizing the runtime dependency story for the legacy training stack under `uv`.
- Blocked:
  - Full dependency resolution is constrained by older `tensorflow` and `tensorforce` compatibility.
- Next week:
  - Smoke-test the main project code against the converted dataset.
  - Continue closing the paper-to-code gaps in training and evaluation.
- Links:
  - `preprocessing/databento/`
  - `data/sample/`
  - `data/validation/`

# Contributor Update: Anja
- Wrote full simplified pipeline in `paper_replication.ipynb`

# Contributor Update: Pierre
- uploaded all data to cluster (currently in scratch as no dedicated storage found)
- Started writing complete paper (continuous part only) replication pipeline in PyTorch in `lobmm/`
- Warning: data normalization is still different than in the paper. (log instead of divide by max volume, no perfect stationarity, but assume paper doesn't have it either)


## Week of 2026-03-16

### Weekly Snapshot

TODO
