import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/human_control"))
import annotate_control_rq2_lifecycle as pipeline

BASE = ROOT / "src/output/human_control/final_matching_current"
PAIRS = BASE / "matched_pairs.json"
CLUSTER = BASE / "control_ctd_clustering.jsonl"
INPUT = BASE / "control_rq2_lifecycle_current_input.jsonl"

pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
pair_map = {
    f'{p["control"]["repo"]}#{p["control"]["number"]}': f'{p["ai"]["repo"]}#{p["ai"]["number"]}'
    for p in pairs
}
control_keys = set(pair_map)
rows = []
for line in CLUSTER.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row.get("pr_key") in control_keys and row.get("valid") is True:
        rows.append({"cohort": "CONTROL", "pr_key": row["pr_key"], "valid": True, "judgment": row["judgment"]})
        ai_key = pair_map[row["pr_key"]]
        rows.append({"cohort": "AI", "pr_key": ai_key, "valid": True, "judgment": {"obligations": [{}]}})

INPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
pipeline.PAIRS = PAIRS
pipeline.JUDGMENT = INPUT
pipeline.SZZ_ROOT = BASE / "control_ctd_clustering_input"
pipeline.REPO_AUDIT = ROOT / "src/output/full_scale/rq2_control_repository_evidence.591.json"
pipeline.OUT_ROOT = BASE / "control_rq2_lifecycle_current"
pipeline.main()
