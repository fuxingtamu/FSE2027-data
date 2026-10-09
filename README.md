# FSE2027-data

## RQ1 human annotation: PR purpose (500 cases)

Coder A and Coder B each labeled 500 PRs, with all case IDs aligned. The comparison below includes the GPT-5.6-Luna reference labels. Agreement is descriptive for these saved annotations and should not be treated as independent-coder reliability because the annotation workflow included model references and human resolutions.

| Pair | Agreement | Cohen's unweighted κ |
| --- | ---: | ---: |
| Coder A vs Coder B | 492/500 (98.4%) | 0.9796 |
| Coder A vs GPT-5.6-Luna | 489/500 (97.8%) | 0.9718 |
| Coder B vs GPT-5.6-Luna | 487/500 (97.4%) | 0.9668 |

There are 8 Coder A/B disagreements: PUR-048, PUR-081, PUR-097, PUR-165, PUR-282, PUR-337, PUR-483, and PUR-486. Across all three pairwise comparisons, 16 cases differ in at least one pair. Label counts are:

| Purpose | Coder A | Coder B | GPT-5.6-Luna |
| --- | ---: | ---: | ---: |
| bug-fix | 178 | 175 | 183 |
| feature | 114 | 117 | 116 |
| refactor | 35 | 35 | 34 |
| test | 20 | 22 | 17 |
| documentation | 52 | 53 | 53 |
| configuration-dependency | 59 | 58 | 57 |
| maintenance | 35 | 33 | 33 |
| mixed-purpose | 4 | 4 | 4 |
| other | 3 | 3 | 3 |

These are nominal, unweighted Cohen's κ values. The historical GPT-5.6-Luna comparison is a reference-label comparison, not an independent human reliability measure.

## RQ1 human re-audit: CTD screening and classification (480 cases)

The Coder A and Coder B files each contain 480 matching case IDs. The figures below compare the saved labels and are **descriptive file-to-file agreement**, not an estimate of independent inter-rater reliability: both files were developed through a workflow that consulted the same prior LLM labels and incorporated case-by-case human resolutions.

| Measure | Agreement | Cohen's unweighted κ |
| --- | ---: | ---: |
| CTD presence (`retain_cmo`) | 448/480 (93.3%) | 0.867 |
| Category, among 213 cases both coders marked CTD | 194/213 (91.1%) | 0.892 |
| Subtype, among 213 cases both coders marked CTD | 193/213 (90.6%) | 0.899 |
| Full category label, with non-CTD treated as a class | 429/480 (89.4%) | 0.845 |
| Full subtype label, with non-CTD treated as a class | 428/480 (89.2%) | 0.848 |

The CTD-presence confusion counts are: both CTD 213, both non-CTD 235, A=CTD/B=non-CTD 8, and A=non-CTD/B=CTD 24. Coder A marked 221 cases CTD and Coder B marked 237. Category counts among CTD cases are:

| Category | Coder A | Coder B |
| --- | ---: | ---: |
| TD-1 | 25 | 23 |
| TD-2 | 33 | 33 |
| TD-3 | 70 | 76 |
| TD-4 | 8 | 10 |
| TD-5 | 33 | 40 |
| TD-6 | 5 | 5 |
| TD-7 | 19 | 15 |
| TD-8 | 3 | 4 |
| TD-9 | 6 | 6 |
| TD-10 | 3 | 3 |
| TD-11 | 16 | 22 |

There are 32 CTD-presence disagreements: B-0009, B-0019, B-0024–B-0027, B-0032–B-0033, B-0035, B-0037, B-0044, B-0053, B-0093, B-0103–B-0104, B-0148, B-0170, B-0187, B-0196, B-0226, B-0243, B-0274, B-0300, B-0328, B-0342, B-0349, B-0370, B-0383, B-0395, B-0449, B-0455, and B-0472. Among cases both marked CTD, 19 category labels differ: B-0008, B-0036, B-0041, B-0055, B-0069, B-0146–B-0147, B-0163, B-0183, B-0203, B-0251, B-0284, B-0336, B-0389, B-0404, B-0420, B-0425, B-0434, and B-0471. One additional case, B-0144, has the same category but a different subtype.

**Field and interpretation cautions.** `related` and `retain_cmo` are separate dimensions in the coding guide. In the saved Coder B file, `related` equals `retain_cmo` in all 480 rows, so agreement on `related` is not a valid independent comparison. In B-0301–B-0480, `addition_aware` was mechanically set to the inverse of `retain_cmo`; do not interpret a file-level agreement rate for that field as independent coding agreement. Non-CTD rows have blank category fields in both files; for the full-label calculations above, these were normalized to a single non-CTD class.

## RQ2 human annotation: origin mechanism (120 AI CTDs)

Coder A and Coder B labeled the same 120 AI CTDs using the current eight-class origin-mechanism codebook. This sample consists of rows 1–120, in file order, from `results/RQ2/rq2_ai_lifecycle_labels_8class_1058.jsonl`; it is not a random sample. The case-to-blind-ID mapping is saved in `results/RQ2/rq2_ai_lifecycle_labels_8class_sample120.jsonl`. Both coders saw the new GPT-5.6-Luna labels and rationale, and could adopt either the assistant's assessment or the model label. These file-to-file agreement statistics are descriptive and should not be interpreted as independent inter-rater reliability or as estimates for all 1,058 CTDs.

| Pair | Agreement | Cohen's κ (nominal) |
| --- | ---: | ---: |
| Coder A vs Coder B | 97/120 (80.8%) | 0.771 |
| Coder A vs GPT-5.6-Luna | 102/120 (85.0%) | 0.820 |
| Coder B vs GPT-5.6-Luna | 107/120 (89.2%) | 0.869 |

The 23 Coder A/B disagreements are: ORG-027, ORG-029, ORG-032, ORG-037–ORG-039, ORG-051–ORG-052, ORG-055–ORG-056, ORG-058, ORG-060, ORG-064, ORG-067, ORG-074, ORG-081, ORG-084, ORG-091, ORG-101, ORG-107–ORG-108, ORG-110, and ORG-112. Coder A differs from GPT-5.6-Luna on 18 cases: ORG-001, ORG-011, ORG-027, ORG-029, ORG-032, ORG-037, ORG-051–ORG-052, ORG-054, ORG-064, ORG-066–ORG-067, ORG-074, ORG-101, ORG-107–ORG-108, ORG-110, and ORG-112. Coder B differs from GPT-5.6-Luna on 13 cases: ORG-001, ORG-011, ORG-038–ORG-039, ORG-054–ORG-056, ORG-058, ORG-060, ORG-066, ORG-081, ORG-084, and ORG-091.

| Origin mechanism | Coder A | Coder B | GPT-5.6-Luna |
| --- | ---: | ---: | ---: |
| `IMPLEMENTATION_SHORTCUT` | 20 | 24 | 34 |
| `CONTEXT_EVOLUTION` | 18 | 19 | 19 |
| `CONTRACT_ASSUMPTION_MISMATCH` | 25 | 23 | 24 |
| `VERIFICATION_FEEDBACK_GAP` | 25 | 24 | 22 |
| `STRUCTURAL_CONSTRAINT` | 18 | 14 | 7 |
| `SCOPE_REQUIREMENT_OMISSION` | 10 | 11 | 9 |
| `UNKNOWN` | 3 | 4 | 4 |
| `EMERGENT_CONTEXT_CHANGE` | 1 | 1 | 1 |

The saved annotation files omit confidence fields. Coder A selected the assistant assessment for 18 cases and the GPT-5.6-Luna label for 102; Coder B selected the assistant assessment for 13 cases and the GPT-5.6-Luna label for 107.

## RQ2 human annotation: exposure evidence (120 PRs)

Coder A and Coder B labeled the same 120 PRs for exposure family and subtype. Coder B’s completed file was synchronized from the generation directory into `section25_human_reaudit/`; four semantically equivalent subtype strings were standardized to the official codebook spellings before calculating normalized subtype agreement: `EX1.1_ISSUE_BUG_REPORT`, `EX3.2_REQUIREMENT_FEATURE_EVOLUTION`, `EX3.3_STATE_DATA_SCALE_EVOLUTION`, and `EX4.1_TEST_COVERAGE_TOOLING`. Raw string agreement is also reported for transparency.

| Pair / level | Agreement | Cohen's κ (nominal) |
| --- | ---: | ---: |
| Coder A vs Coder B, exposure family | 103/120 (85.8%) | 0.766 |
| Coder A vs Coder B, raw subtype strings | 98/120 (81.7%) | 0.796 |
| Coder A vs Coder B, subtype after code normalization | 100/120 (83.3%) | 0.813 |
| Coder A vs evidence-reviewed model labels, unambiguous PR subset | 36/44 (81.8%) | 0.788 |
| Coder B vs evidence-reviewed model labels, unambiguous PR subset | 40/44 (90.9%) | 0.891 |

The model comparison uses `src/output/full_scale/rq2_exposure_evidence_subtypes_final.json` and is limited to 44 PRs whose associated CTD-level records all have the same final subtype. The other 76 sampled PRs have multiple subtype labels among their associated CTDs, so they do not have a single directly comparable model label at PR level. After normalizing equivalent code spellings, the 20 Coder A/B subtype disagreements are EXP-012, EXP-030, EXP-041, EXP-043, EXP-044, EXP-047, EXP-052, EXP-061, EXP-065, EXP-081, EXP-084, EXP-087, EXP-101, EXP-102, EXP-109, EXP-110, EXP-111, EXP-114, EXP-116, and EXP-119. Family counts (EX1/EX2/EX3/EX4/EXU) are 15/7/66/30/2 for Coder A and 11/7/70/28/4 for Coder B. The finalized coder files are `section25_human_reaudit/exposure_120_coder_A.jsonl` and `_coder_B.jsonl` (CSV versions are also provided); the original generated Coder B strings remain in `src/output/full_scale/section25_human_reaudit/`. These are descriptive agreement statistics, not independent reliability estimates, because the annotation workflow included shared model references and human resolutions.

## RQ3 human annotation: LLM issue-to-CTD matching (120 issues)

Each issue can be matched to zero, one, or several CTDs. The table reports exact agreement on the full set of matched CTD indices, plus Cohen's κ for the binary case-level decision of whether an issue matches any CTD. The GPT reference is the protected prior model result for this sample.

| Pair | Exact matched-set agreement | Any-match agreement | Cohen's κ (any match) |
| --- | ---: | ---: | ---: |
| Coder A vs Coder B | 110/120 (91.7%) | 111/120 (92.5%) | 0.849 |
| Coder A vs GPT reference | 111/120 (92.5%) | 112/120 (93.3%) | 0.867 |
| Coder B vs GPT reference | 113/120 (94.2%) | 113/120 (94.2%) | 0.883 |

When expanded to the 305 issue–CTD decisions, agreement / nominal Cohen's κ is 290/305 (95.1%) / 0.853 for Coder A vs Coder B, 292/305 (95.7%) / 0.876 for Coder A vs GPT, and 297/305 (97.4%) / 0.925 for Coder B vs GPT. There were no UNCERTAIN per-index labels in the saved coder files. Case-level positive counts are 56 for Coder A, 55 for Coder B, and 60 for GPT. Exact matched-set disagreements are: A/B — LI-016, LI-039, LI-061, LI-063, LI-077, LI-078, LI-093, LI-101, LI-107, and LI-115; A/GPT — LI-016, LI-038, LI-039, LI-041, LI-063, LI-077, LI-078, LI-098, and LI-101; B/GPT — LI-038, LI-041, LI-061, LI-093, LI-098, LI-107, and LI-115. These comparisons are descriptive rather than independent inter-rater reliability because the coders consulted model references during annotation. The coder files are `section25_human_reaudit/llm_issue_match_120_coder_A.jsonl` and `_coder_B.jsonl` (CSV versions are also provided).

## RQ3 human annotation: static finding-to-CTD matching (200 findings)

Coder A and Coder B independently reviewed the same 200 static-analysis findings against their associated CTD repair. A finding is labeled YES only when it identifies the same underlying issue addressed by the repair; removing a file or changing unrelated code does not by itself count as a match. GPT-5.6-Luna labels are included as a reference. These are descriptive file-to-file agreement statistics, not independent reliability estimates, because both annotators consulted the model results and resolved cases through this assisted workflow.

| Pair | Agreement | Cohen's κ (nominal) |
| --- | ---: | ---: |
| Coder A vs Coder B | 186/200 (93.0%) | 0.859 |
| Coder A vs GPT-5.6-Luna | 193/200 (96.5%) | 0.930 |
| Coder B vs GPT-5.6-Luna | 191/200 (95.5%) | 0.910 |

The YES/NO counts are 93/107 for Coder A, 93/107 for Coder B, and 100/100 for GPT-5.6-Luna. The 14 Coder A/B disagreements are SA-056, SA-068, SA-100, SA-102, SA-107, SA-111, SA-125, SA-126, SA-153, SA-154, SA-156, SA-178, SA-185, and SA-199. Coder A differs from GPT-5.6-Luna on SA-093, SA-102, SA-107, SA-125, SA-154, SA-185, and SA-199. Coder B differs from GPT-5.6-Luna on SA-056, SA-068, SA-093, SA-100, SA-111, SA-126, SA-153, SA-156, and SA-178.

The completed Coder B annotation is saved in `section25_human_reaudit/static_match_200_coder_B.jsonl` and `.csv`; the corresponding Coder A files use the same names with `coder_A`.

Research data and analysis scripts for a study of AI-attributed pull requests (PRs), matched control PRs, and technical debt (TD). The repository contains PR rosters, matched pairs, SZZ trace summaries, classification labels, lifecycle annotations, and review and static-analysis judgments. Files are organized by research question (RQ); the remainder of this README inventories the published artifacts.

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

The published data and result files can be read with Python's standard library as shown above. Executing the full pipeline requires additional dependencies, API credentials, source-repository data, and intermediate files referenced by the scripts; these are not all included here. Some scripts use paths from the original research environment. Review each script's arguments, configured paths, and external-service calls before running it. Set credentials through environment variables such as `GITHUB_TOKEN` and the relevant LLM API key variables; do not commit credentials or local `.env` files.

## Citation and license

If you use this repository, cite the associated paper when its bibliographic details are available and link to this repository. No license file is included; contact the repository owner about reuse terms.
