"""Evidence-only exposure subtype review for the strict Control cohort."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/output/full_scale"
sys.path.insert(0, str(ROOT / "src/phase2_szz"))
import td_classifier_obligation as llm  # noqa: E402

INPUT = OUT / "rq2_control_lifecycle_calibrated_finalrule.json"
ALLOWED = {
    "EX1_ISSUE_OR_BUG_REPORT", "EX1_USER_VISIBLE_REPORT", "EX1_OTHER_EXPLICIT_REPORT",
    "EX2_CI_FAILURE", "EX2_TEST_FAILURE", "EX2_STATIC_OR_LINT_SIGNAL", "EX2_OTHER_AUTOMATED_SIGNAL",
    "EX3_CALLER_OR_INTERFACE_EVOLUTION", "EX3_REQUIREMENT_OR_FEATURE_EVOLUTION",
    "EX3_STATE_DATA_OR_SCALE_EVOLUTION", "EX3_DEPENDENCY_OR_PLATFORM_EVOLUTION", "EX3_OTHER_CODE_EVOLUTION",
    "EX4_TEST_COVERAGE_OR_TOOLING", "EX4_REFACTOR_OR_CLEANUP",
    "EX4_STYLE_DOC_CONFIG_MAINTENANCE", "EX4_OTHER_INTERNAL_MAINTENANCE",
    "EXU_INSUFFICIENT_EVIDENCE",
}

SYSTEM = """Classify only the primary observable exposure-evidence subtype for this already-coded Control CMO.
Do not change origin mechanism, temporal status, pathway, repair response, or TD category. Use only the supplied
evidence. An issue reference alone is not a failure; a test-file edit alone is not a test failure. Choose exactly one
allowed subtype. Use EXU_INSUFFICIENT_EVIDENCE only when the evidence does not support any specific subtype.
Return JSON only: {\"subtype\":\"...\",\"confidence\":\"high|medium|low\",\"evidence\":\"brief reason\"}."""

CURRENT_RESULTS = ROOT / "src/output/human_control/final_matching_current/control_rq2_lifecycle_current/results.json"
CURRENT_SEED = OUT / "rq2_control_exposure_evidence_subtypes_review.jsonl"
CURRENT_OUTPUT = ROOT / "src/output/human_control/final_matching_current/control_exposure_subtypes_current.jsonl"


def prompt(row: dict) -> str:
    return "\n".join([
        f"TD category: {row.get('category')}",
        f"Existing pathway (context only): {row.get('exposure_pathway')}",
        f"Temporal status (context only): {row.get('temporal_status')}",
        f"Origin mechanism (context only): {row.get('origin_mechanism')}",
        f"Repair response (context only): {row.get('repair_response')}",
        f"Allowed subtypes: {sorted(ALLOWED)}",
        "Evidence:", row.get("evidence", ""),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--output", default=str(OUT / "rq2_control_exposure_evidence_subtypes_review.jsonl"))
    ap.add_argument("--current", action="store_true", help="review missing subtypes for the current fixed control CTDs")
    args = ap.parse_args()
    if args.current:
        current = json.loads(CURRENT_RESULTS.read_text(encoding="utf-8"))
        rows = []
        for pr in current:
            if pr.get("valid") is not True:
                continue
            for ob in pr.get("judgment", {}).get("obligations", []):
                rows.append({
                    "obligation_id": f"{pr['pr_key']}:{ob['obligation_id']}",
                    "exposure_pathway": ob.get("exposure_pathway"),
                    "temporal_status": "YES",
                    "origin_mechanism": ob.get("origin_mechanism"),
                    "repair_response": ob.get("repair_response"),
                    "category": None,
                    "evidence": ob.get("evidence", ""),
                })
        output_path = Path(args.output) if args.output != str(OUT / "rq2_control_exposure_evidence_subtypes_review.jsonl") else CURRENT_OUTPUT
    else:
        rows = json.loads(INPUT.read_text(encoding="utf-8"))["rows"]
        output_path = Path(args.output)
    output = output_path
    done = {}
    if args.current and CURRENT_SEED.exists():
        current_ids = {str(r["obligation_id"]) for r in rows}
        for line in CURRENT_SEED.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if str(item.get("obligation_id")) in current_ids and item.get("valid") is True:
                    done[str(item["obligation_id"])] = item
            except (ValueError, KeyError):
                pass
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                done[str(item["obligation_id"])] = item
            except (ValueError, KeyError):
                pass
    todo = [r for r in rows if str(r["obligation_id"]) not in done]
    print(f"rows={len(rows)} todo={len(todo)}", flush=True)

    def one(row: dict) -> dict:
        raw = llm.call_llm(SYSTEM, prompt(row), model=args.model, max_retries=6, max_tokens=700)
        valid = isinstance(raw, dict) and raw.get("subtype") in ALLOWED and raw.get("confidence") in {"high", "medium", "low"}
        return {
            "obligation_id": row["obligation_id"],
            "exposure_pathway": row.get("exposure_pathway"),
            "review": raw,
            "valid": valid,
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, output.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(one, row) for row in todo]
        for i, future in enumerate(as_completed(futures), 1):
            item = future.result()
            done[str(item["obligation_id"])] = item
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] valid={item['valid']}", flush=True)
    if args.current:
        output.write_text("\n".join(json.dumps(done[k], ensure_ascii=False) for k in sorted(done)) + "\n", encoding="utf-8")
    summary = {
        "n": len(done),
        "valid": sum(bool(x.get("valid")) for x in done.values()),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
