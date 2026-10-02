# FSE2027-data

Research data and analysis scripts for a study of AI-attributed pull requests (PRs), matched control PRs, and technical debt (TD). The repository contains PR rosters, matched pairs, SZZ trace summaries, classification labels, lifecycle annotations, and review and static-analysis judgments. Files are organized by research question (RQ); this README describes the published artifacts without interpreting the study's findings.

## Repository contents

| Path | Contents | Records |
| --- | --- | ---: |
| [`data/ai_prs_2717.csv`](data/ai_prs_2717.csv) and [JSON](data/ai_prs_2717.json) | AI-attributed PR roster; repository, PR number, agent, dates, source, and change-size metadata | 2,717 PRs |
| [`data/matched_ai_control_prs_801.csv`](data/matched_ai_control_prs_801.csv) and [JSON](data/matched_ai_control_prs_801.json) | AI/control PR pairs and matching metadata | 801 pairs |
| [`results/shared/szz/`](results/shared/szz/) | Per-repository compressed JSON Lines summaries for AI and control cohorts | 20 repositories per cohort |
| [`results/RQ1/`](results/RQ1/) | PR-purpose labels and CTD category/subtype labels | See below |
| [`results/RQ2/`](results/RQ2/) | CTD lifecycle and exposure annotations | See below |
| [`results/RQ3/`](results/RQ3/) | Issue-level, LLM-review, and static-finding judgments | See below |
| [`scripts/`](scripts/) | Collection, matching, SZZ, CTD, and RQ1–RQ3 processing scripts | 19 Python files |

### Result files

Each `.jsonl` file contains one JSON object per line. The counts below are **rows in each file**, not necessarily unique PRs or unique CTDs; the files represent different analysis units and should not be combined by row position.

| File | Rows | Unit / key fields |
| --- | ---: | --- |
| `results/RQ1/purpose_ai_2717.jsonl` | 2,717 | AI PR; `repo`, `pr_number`, `purpose` |
| `results/RQ1/purpose_matched_control_801.jsonl` | 801 | Control PR; `repo`, `pr_number`, `purpose` |
| `results/RQ1/ai_ctd_labels_1058.jsonl` | 1,058 | AI CTD; `ctd_key`, `category`, `subtype` |
| `results/RQ1/matched_ai_control_ctd_labels.jsonl` | 414 | CTD in the matched AI/control analysis; `ctd_key`, `cohort` |
| `results/RQ2/rq2_ai_lifecycle_labels.jsonl` | 1,044 | AI CTD lifecycle annotation; `ctd_key` |
| `results/RQ2/rq2_matched_control_lifecycle_labels.jsonl` | 160 | Control CTD lifecycle annotation; `ctd_key` |
| `results/RQ3/rq3_issue_level_judgments.jsonl` | 1,848 | Review issue at a replay level; `pr_key`, `level`, `issue_index` |
| `results/RQ3/rq3_llm_review_judgments.jsonl` | 2,655 | PR review at a replay level; `pr_key`, `level` |
| `results/RQ3/rq3_static_finding_judgments.jsonl` | 101,207 | Individual static-analysis diagnostic; `finding_index`, `target_ctd_index`, `tool` |

The RQ1 and RQ2 CTD files have different coverage. In particular, the 1,044 RQ2 AI lifecycle rows are not a one-to-one copy of the 1,058 RQ1 AI CTD rows. Use `ctd_key` for CTD-level joins and inspect cohort and scope before comparing files. RQ3 index fields use the scope recorded in each judgment; do not assume they are interchangeable with the `ctd_index` within a PR in RQ1.

## Reading the data

CSV files have headers and UTF-8 text. The JSON roster files contain arrays of objects. JSONL files contain one object per line; SZZ summaries use gzip-compressed JSONL (`.jsonl.gz`). For example, from the repository root:

```python
import csv
import gzip
import json
from pathlib import Path

with Path("data/ai_prs_2717.csv").open(encoding="utf-8-sig", newline="") as f:
    ai_prs = list(csv.DictReader(f))

with Path("results/RQ1/ai_ctd_labels_1058.jsonl").open(encoding="utf-8-sig") as f:
    ctd_labels = [json.loads(line) for line in f if line.strip()]

with gzip.open("results/shared/szz/ai/bun.jsonl.gz", "rt", encoding="utf-8-sig") as f:
    szz_cases = [json.loads(line) for line in f if line.strip()]
```

CSV values are read as strings; convert numeric columns explicitly for analysis. A PR is identified by repository and PR number. Matched-pair rows use `ai_repo`/`ai_number` and `control_repo`/`control_number`; result files generally use `repo`/`pr_number` or a composite `pr_key`. SZZ files are partitioned by cohort and repository, and their `verdict_level` and `szz_confidence` fields describe trace assessments rather than confirmed TD by themselves.

## Scripts and reproducibility

The scripts are grouped by stage:

1. `scripts/collect_and_match_controls.py` collects control candidates and builds matched pairs.
2. `scripts/szz/` traces and enriches subsequent changes.
3. `scripts/ctd/` classifies and clusters CTD cases.
4. `scripts/rq1/`, `scripts/rq2/`, and `scripts/rq3/` generate the research-question annotations and judgments.

The published data and result files can be read with Python's standard library as shown above. Rerunning the full pipeline requires additional dependencies, API credentials, source-repository data, and intermediate files referenced by the scripts; these are not all included here. Some scripts use paths from the original research environment. Review each script's arguments, configured paths, and external-service calls before running it. Set credentials through environment variables such as `GITHUB_TOKEN` and the relevant LLM API key variables; do not commit credentials or local `.env` files.

## Citation and license

If you use this repository, cite the associated paper when its bibliographic details are available and link to this repository. No license file is included; contact the repository owner about reuse terms.
