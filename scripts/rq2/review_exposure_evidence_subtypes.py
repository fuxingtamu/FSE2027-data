"""Evidence-only LLM review for the second-level exposure subtype.

This review never changes origin mechanism, temporal status, pathway, or repair
response. It only adjudicates the subtype nested under the existing EX family.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "src/output/full_scale"
sys.path.insert(0, str(ROOT / "src/phase2_szz"))
import td_classifier_obligation as llm  # noqa: E402

PACKETS = OUT_ROOT / "rq2_full_obligation_packets.json"
PROVISIONAL = OUT_ROOT / "rq2_exposure_evidence_subtypes_provisional.json"

ALLOWED = {
    "EX1.1_ISSUE_OR_BUG_REPORT", "EX1.2_USER_VISIBLE_REPORT", "EX1.4_OTHER_EXPLICIT_REPORT",
    "EX2.1_CI_FAILURE", "EX2.2_TEST_FAILURE", "EX2.3_STATIC_OR_LINT_SIGNAL", "EX2.5_OTHER_AUTOMATED_SIGNAL",
    "EX3.1_CALLER_OR_INTERFACE_EVOLUTION", "EX3.2_REQUIREMENT_OR_FEATURE_EVOLUTION",
    "EX3.3_STATE_DATA_OR_SCALE_EVOLUTION", "EX3.4_DEPENDENCY_OR_PLATFORM_EVOLUTION", "EX3.5_OTHER_CODE_EVOLUTION",
    "EX4.1_TEST_COVERAGE_OR_TOOLING", "EX4.2_REFACTOR_OR_CLEANUP",
    "EX4.3_STYLE_DOC_CONFIG_MAINTENANCE", "EX4.5_OTHER_INTERNAL_MAINTENANCE",
    "EXU.1_UNSUFFICIENT_EVIDENCE",
}

SYSTEM = """You classify only the primary exposure-evidence subtype for one already-coded CMO.
Do not change or infer origin mechanism, temporal status, exposure pathway, TD category, or repair response.
Use only the supplied origin PR, linked issue, later repair commit, and repository evidence.
Assign exactly one subtype from the allowed list. A keyword alone is not sufficient: require an explicit
failure/report/CI/test/tool/evolution/maintenance cue. If the evidence does not support a subtype, use the
family's OTHER subtype; use EXU.1_UNSUFFICIENT_EVIDENCE only when the existing EX family is EXU.
Return JSON only: {\"exposure_evidence_subtype\":\"...\",\"confidence\":\"high|medium|low\",\"evidence\":\"brief quoted/paraphrased evidence\"}."""


def short(value: object, limit: int = 4500) -> str:
    s = str(value or "")
    return s if len(s) <= limit else s[:limit] + " ...[truncated]..."


def packet_text(packet: dict) -> str:
    parts = []
    for m in packet.get("members", []):
        parts += [
            f"FILE: {m.get('filepath')}",
            f"PR TITLE: {m.get('pr_title')}",
            "PR BODY:\n" + short(m.get("pr_body"), 4500),
            f"REPAIR COMMIT: {m.get('commit_message')}",
            "LINKED ISSUES:\n" + short(m.get("linked_issues"), 2500),
            f"ISSUE REFS FOUND: {m.get('issue_refs_found')}",
            "REPAIR DIFF:\n" + short((m.get("fix_diff") or {}).get("diff"), 4500),
        ]
    return "\n\n".join(parts)


def prompt(row: dict, packet: dict) -> str:
    return "\n".join([
        f"Existing evidence family: {row['exposure']}",
        f"Existing lifecycle pathway (context only): {row.get('pathway')}",
        f"Allowed subtypes: {sorted(ALLOWED)}",
        "EVIDENCE PACKET:", packet_text(packet),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--checkpoint", default=str(OUT_ROOT / "rq2_exposure_evidence_subtypes_review.jsonl"))
    ap.add_argument("--output", default=str(OUT_ROOT / "rq2_exposure_evidence_subtypes_review.json"))
    args = ap.parse_args()
    provisional = json.loads(PROVISIONAL.read_text(encoding="utf-8"))["rows"]
    packets = json.loads(PACKETS.read_text(encoding="utf-8"))["obligations"]
    packet_map = {int(p["obligation_id"]): p for p in packets}
    candidates = [r for r in provisional if r["subtype_confidence"] != "high"]
    if args.limit:
        # deterministic stratification across parent families and provisional labels
        groups = {}
        for r in candidates:
            groups.setdefault(r["exposure"], []).append(r)
        selected = []
        families = sorted(groups)
        cursor = 0
        while len(selected) < min(args.limit, len(candidates)):
            fam = families[cursor % len(families)]
            if groups[fam]: selected.append(groups[fam].pop(0))
            cursor += 1
            if cursor > len(families) * len(candidates) + 1: break
        candidates = selected
    checkpoint = Path(args.checkpoint)
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                x = json.loads(line); done[str(x["obligation_id"])] = x
            except (ValueError, KeyError):
                pass
    todo = [r for r in candidates if str(r["obligation_id"]) not in done]
    print(f"candidates={len(candidates)} todo={len(todo)}", flush=True)

    def one(row: dict) -> dict:
        raw = llm.call_llm(SYSTEM, prompt(row, packet_map[int(row["obligation_id"])]), model=args.model, max_retries=6, max_tokens=1000)
        ok = isinstance(raw, dict) and raw.get("exposure_evidence_subtype") in ALLOWED and raw.get("confidence") in {"high", "medium", "low"}
        return {"obligation_id": row["obligation_id"], "exposure": row["exposure"], "provisional_subtype": row["exposure_evidence_subtype"], "review": raw, "valid": ok}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, checkpoint.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(one, r) for r in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            x = fut.result(); done[str(x["obligation_id"])] = x
            fh.write(json.dumps(x, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{i}/{len(todo)}] id={x['obligation_id']} valid={x['valid']}", flush=True)
    output = Path(args.output)
    output.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"reviewed": len(done), "valid": sum(x.get('valid') is True for x in done.values()), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
