"""Re-annotate the full AI CTD lifecycle set with the current RQ2 codebook.

The model sees only the source PR, repair, issue, and repository evidence.
Legacy labels and rationales are attached after inference for audit/comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("D:/ai_td_experiments")
REPO = Path(__file__).resolve().parents[2]
RQ1 = REPO / "results/RQ1/ai_ctd_labels_1058.jsonl"
RQ2 = REPO / "results/RQ2/rq2_ai_lifecycle_labels.jsonl"
PACKETS = ROOT / "src/output/full_scale/rq2_full_obligation_packets.json"
MODEL_ANNOTATIONS = ROOT / "src/output/full_scale/rq2_full_annotations_both.json"
OUT = ROOT / "src/output/full_scale/ai_ctd_lifecycle_8class_1058.jsonl"
CHECKPOINT = ROOT / "src/output/full_scale/ai_ctd_lifecycle_8class_1058.checkpoint.jsonl"
SUMMARY = ROOT / "src/output/full_scale/ai_ctd_lifecycle_8class_1058_summary.json"
sys.path.insert(0, str(ROOT / "src/phase2_szz"))
import td_classifier_obligation as llm  # noqa: E402

MODEL = "gpt-5.6-luna"
CHUNK_SIZE = 5
MAX_PROMPT_CHARS = 120_000

# Uses the current Control lifecycle codebook and strengthened pathway rules.
# Prior labels, root-cause rationales, and TD subtypes are intentionally hidden.
SYSTEM = r"""You are annotating technical-debt lifecycle cases for a software-engineering study.
Use only the supplied origin PR, later repair, linked issue/CI, and repository evidence.
Do not use authorship, cohort, TD category/subtype, or any prior lifecycle label as evidence.
Return JSON only.

For each supplied case, return exactly one object with:
- origin_mechanism: SCOPE_REQUIREMENT_OMISSION, IMPLEMENTATION_SHORTCUT,
  CONTRACT_ASSUMPTION_MISMATCH, STRUCTURAL_CONSTRAINT,
  VERIFICATION_FEEDBACK_GAP, CONTEXT_EVOLUTION, EMERGENT_CONTEXT_CHANGE,
  or UNKNOWN. Explain why the obligation was left unresolved at origin; do not copy a TD subtype.
- temporal_status: YES if applicable at origin merge, NO if applicable only after later evolution,
  UNCERTAIN otherwise.
- exposure_pathway: P-A only for explicit failure/user report/issue/CI/test/static-analysis evidence;
  P-B only when a later production-code/interface/caller/requirement change made an existing
  obligation observable; P-C for test expansion, lint/style cleanup, refactoring, deletion,
  or internal maintenance without either trigger; P-U if evidence cannot distinguish routes.
  Apply this order: P-A, then P-B, then P-C, then P-U. Do not infer P-A from a commit title alone.
- repair_response: RP1 guard/check/error handling; RP2 logic/contract/type correction;
  RP3 refactoring/deletion; RP4 style/documentation/convention; RP5 test/tooling;
  RP6 dependency/configuration.
- confidence: high, medium, or low.
- evidence: concise file/commit evidence for all decisions.

If supplied evidence cannot establish origin applicability or a specific mechanism, use
UNCERTAIN and/or UNKNOWN as appropriate. Return {"obligations":[{"obligation_id":"O1",
"origin_mechanism":...,"temporal_status":...,"exposure_pathway":...,"repair_response":...,
"confidence":...,"evidence":...}]} exactly."""

MECHANISMS = {
    "SCOPE_REQUIREMENT_OMISSION", "IMPLEMENTATION_SHORTCUT",
    "CONTRACT_ASSUMPTION_MISMATCH", "STRUCTURAL_CONSTRAINT",
    "VERIFICATION_FEEDBACK_GAP", "CONTEXT_EVOLUTION",
    "EMERGENT_CONTEXT_CHANGE", "UNKNOWN",
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(row: dict, packet: bool = False) -> tuple[str, str, str, str]:
    pr = row.get("pr") if packet else row.get("pr_number")
    return (str(row["repo"]), str(pr), str(row["category"]), str(row["subtype"]))


def short(value: object, limit: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "\n... [truncated] ..."


def prepare() -> tuple[list[dict], dict[int, dict], dict[str, dict]]:
    ctds = jsonl(RQ1)
    legacy = {row["ctd_key"]: row for row in jsonl(RQ2)}
    packets = json.loads(PACKETS.read_text(encoding="utf-8"))["obligations"]
    old_model = json.loads(MODEL_ANNOTATIONS.read_text(encoding="utf-8"))
    packet_by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    ctd_by_key: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for packet in packets:
        packet_by_key[group_key(packet, packet=True)].append(packet)
    for ctd in ctds:
        ctd_by_key[group_key(ctd)].append(ctd)
    for rows in packet_by_key.values():
        rows.sort(key=lambda row: int(row["obligation_id"]))
    for rows in ctd_by_key.values():
        rows.sort(key=lambda row: int(row["ctd_index"]))

    if len(ctds) != 1058 or len({row["ctd_key"] for row in ctds}) != 1058:
        raise RuntimeError(f"Expected 1,058 unique RQ1 AI CTDs, got {len(ctds)}")
    if len(legacy) != 1044 or not set(legacy).issubset({row["ctd_key"] for row in ctds}):
        raise RuntimeError("RQ2 lifecycle file must contain 1,044 keys drawn from the RQ1 CTD frame")
    if len(old_model) != len(packets) or len({int(row["obligation_id"]) for row in old_model}) != len(old_model):
        raise RuntimeError("Original GPT rationale rows do not uniquely cover the obligation packets")
    model_by_id = {int(row["obligation_id"]): row for row in old_model}

    aligned = []
    for key, rows in ctd_by_key.items():
        candidates = packet_by_key.get(key, [])
        if len(rows) != len(candidates):
            raise RuntimeError(f"Evidence crosswalk mismatch for {key}: CTDs={len(rows)} packets={len(candidates)}")
        for ctd, packet in zip(rows, candidates):
            rationale = model_by_id.get(int(packet["obligation_id"]))
            if rationale is None:
                raise RuntimeError(f"Missing original GPT rationale for obligation {packet['obligation_id']}")
            old = legacy.get(ctd["ctd_key"])
            if old and old.get("origin_code") != rationale.get("root_cause"):
                raise RuntimeError(f"Crosswalk validation failed for {ctd['ctd_key']}: old RC != original GPT RC")
            aligned.append({"ctd": ctd, "packet": packet, "legacy": old, "original_model": rationale})
    if len(aligned) != 1058:
        raise RuntimeError(f"Crosswalk produced {len(aligned)} cases, expected 1,058")
    return aligned, model_by_id, legacy


def prompt_for(pr_key: str, rows: list[dict]) -> str:
    parts = [
        f"PR: {pr_key}",
        f"THE INPUT CONTAINS EXACTLY {len(rows)} CTD CASE(S). Return exactly {len(rows)} output object(s).",
        "CASES (classify each case independently; category is context only):",
        "Use every listed local ID O1, O2, ... exactly once, and do not invent any other IDs.",
        "All origin-member, file, repair, and issue blocks nested under one O# are evidence for that same CTD; never split them into separate cases.",
    ]
    for i, item in enumerate(rows, 1):
        packet = item["packet"]
        parts += [
            f"\nO{i}: category={packet.get('category')}; affected_files={packet.get('affected_files')}",
            "The category is context only and must not determine the cause.",
        ]
        members = packet.get("members", [])
        if len(members) > 12:
            parts.append(f"NOTE: showing 12 representative member records of {len(members)} total records.")
        for member in members[:12]:
            parts += [
                f"\nORIGIN MEMBER {member.get('filepath')} | merge={member.get('pr_merged_at')}",
                "ORIGIN PR BODY:", short(member.get("pr_body"), 1800),
                "ORIGIN PATCH:", "```diff", short(member.get("pr_file_patch"), 3500), "```",
                f"REPAIR COMMIT: {member.get('commit_sha')} | {member.get('commit_date')} | {member.get('commit_message')}",
                "REPAIR DIFF:", "```diff", short((member.get("fix_diff") or {}).get("diff"), 3500), "```",
                f"linked issues: {short(member.get('linked_issues'), 800)}",
                f"issue refs found: {member.get('issue_refs_found')}",
            ]
    return "\n".join(parts)


def validate(raw: dict, ids: set[str]) -> tuple[bool, list[str]]:
    obligations = raw.get("obligations", []) if isinstance(raw, dict) else []
    got = [str(row.get("obligation_id")) for row in obligations]
    errors = []
    if set(got) != ids or len(got) != len(ids):
        errors.append(f"obligation coverage mismatch expected={sorted(ids)} got={sorted(got)}")
    allowed_t = {"YES", "NO", "UNCERTAIN"}
    allowed_p = {"P-A", "P-B", "P-C", "P-U"}
    allowed_r = {f"RP{i}" for i in range(1, 7)}
    for row in obligations:
        for field, allowed in (("origin_mechanism", MECHANISMS), ("temporal_status", allowed_t),
                               ("exposure_pathway", allowed_p), ("repair_response", allowed_r)):
            if row.get(field) not in allowed:
                errors.append(f"{row.get('obligation_id')} invalid {field}={row.get(field)}")
        if row.get("confidence") not in {"high", "medium", "low"}:
            errors.append(f"{row.get('obligation_id')} invalid confidence")
    return not errors, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="dry-run only the first N chunks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    aligned, _, _ = prepare()
    by_pr: dict[str, list[dict]] = defaultdict(list)
    for row in aligned:
        ctd = row["ctd"]
        by_pr[f"{ctd['repo']}#{ctd['pr_number']}"].append(row)
    chunks = []
    for pr_key, rows in sorted(by_pr.items()):
        pr_chunks = []
        pending = []
        for item in rows:
            candidate = pending + [item]
            candidate_prompt = prompt_for(pr_key, candidate)
            if pending and (len(candidate) > CHUNK_SIZE or len(candidate_prompt) > MAX_PROMPT_CHARS):
                pr_chunks.append(pending)
                pending = [item]
            else:
                pending = candidate
        if pending:
            pr_chunks.append(pending)
        for chunk_number, chunk in enumerate(pr_chunks, 1):
            batch_id = f"{pr_key}:chunk{chunk_number}"
            chunk_prompt = prompt_for(pr_key, chunk)
            if len(chunk_prompt) > MAX_PROMPT_CHARS:
                raise RuntimeError(f"Single prompt exceeds {MAX_PROMPT_CHARS} chars: {batch_id} ({len(chunk_prompt)})")
            chunks.append((batch_id, pr_key, chunk, chunk_prompt))
    if args.limit:
        chunks = chunks[:args.limit]
    print(json.dumps({"ctds_in_frame": 1058, "legacy_labeled": 1044,
                      "without_legacy_rq2_row": 14, "pr_count": len(by_pr),
                      "chunks_to_run": len(chunks), "max_prompt_chars": max((len(x[3]) for x in chunks), default=0),
                      "model": MODEL, "dry_run": args.dry_run}, ensure_ascii=False))
    if args.dry_run:
        return

    done = {}
    if CHECKPOINT.exists():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("valid") is True:
                    done[row["batch_id"]] = row
            except (ValueError, KeyError):
                continue
    todo = [chunk for chunk in chunks if chunk[0] not in done]
    print(f"checkpoint_valid={len(done)} todo={len(todo)}", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()

    def one(chunk):
        batch_id, pr_key, rows, prompt = chunk
        raw = llm.call_llm(SYSTEM, prompt, model=MODEL, max_retries=6, max_tokens=5000)
        ids = {f"O{i}" for i in range(1, len(rows) + 1)}
        ok, errors = validate(raw, ids) if isinstance(raw, dict) and "error" not in raw else (False, ["LLM error"])
        return {"batch_id": batch_id, "pr_key": pr_key, "ctd_keys": [x["ctd"]["ctd_key"] for x in rows],
                "n_cases": len(rows), "prompt_chars": len(prompt), "judgment": raw,
                "valid": ok, "validation_errors": errors}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, CHECKPOINT.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(one, chunk) for chunk in todo]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            with lock:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
            print(f"[{i}/{len(todo)}] {result['batch_id']} valid={result['valid']}", flush=True)

    completed = dict(done)
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if row.get("valid") is True:
                completed[row["batch_id"]] = row
        except (ValueError, KeyError):
            continue
    if len(completed) != len(chunks):
        raise RuntimeError(f"Only {len(completed)}/{len(chunks)} chunks have valid results; checkpoint retained for resume")

    final = []
    for batch_id, pr_key, rows, _ in chunks:
        result = completed[batch_id]
        judgments = {str(x["obligation_id"]): x for x in result["judgment"]["obligations"]}
        for i, item in enumerate(rows, 1):
            ctd = item["ctd"]
            original = item["original_model"]
            final.append({
                "ctd_key": ctd["ctd_key"], "repo": ctd["repo"], "pr_number": ctd["pr_number"],
                "ctd_index": ctd["ctd_index"], "category": ctd["category"], "subtype": ctd["subtype"],
                "new_codebook_version": "RQ2-origin-8class-current",
                "new_judgment": judgments[f"O{i}"],
                "legacy_rq2_labels": item["legacy"],
                "original_gpt_model": "gpt-5.6-luna",
                "original_gpt_rationale": {k: original.get(k) for k in
                    ("root_cause", "root_cause_rationale", "exposure", "exposure_rationale", "repair_response", "repair_rationale")},
                "evidence_packet_obligation_id": item["packet"]["obligation_id"],
                "crosswalk_method": "exact repo+PR+category+subtype group; sorted ctd_index to obligation_id; validated against old RC labels for all 1,044 legacy rows",
            })
    final.sort(key=lambda row: (row["repo"], int(row["pr_number"]), int(row["ctd_index"])))
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final), encoding="utf-8")
    summary = {
        "status": "complete_model_reannotation_not_human_adjudicated",
        "codebook": "RQ2 current eight-mechanism origin codebook; Control lifecycle rules",
        "model": MODEL,
        "n_ctd_rows": len(final),
        "legacy_rq2_rows": sum(row["legacy_rq2_labels"] is not None for row in final),
        "newly_covered_rows": sum(row["legacy_rq2_labels"] is None for row in final),
        "unique_ctd_keys": len({row["ctd_key"] for row in final}),
        "new_origin_mechanism_counts": dict(Counter(row["new_judgment"]["origin_mechanism"] for row in final)),
        "new_temporal_status_counts": dict(Counter(row["new_judgment"]["temporal_status"] for row in final)),
        "new_exposure_pathway_counts": dict(Counter(row["new_judgment"]["exposure_pathway"] for row in final)),
        "new_repair_response_counts": dict(Counter(row["new_judgment"]["repair_response"] for row in final)),
        "new_confidence_counts": dict(Counter(row["new_judgment"]["confidence"] for row in final)),
        "legacy_origin_mechanism_counts": dict(Counter(row["legacy_rq2_labels"]["origin_mechanism"] for row in final if row["legacy_rq2_labels"])),
        "legacy_origin_code_counts": dict(Counter(row["legacy_rq2_labels"]["origin_code"] for row in final if row["legacy_rq2_labels"])),
        "outputs": {"labels_jsonl": str(OUT), "checkpoint_jsonl": str(CHECKPOINT)},
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": str(SUMMARY), "rows": len(final),
                      "legacy_rq2_rows": summary["legacy_rq2_rows"],
                      "newly_covered_rows": summary["newly_covered_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
