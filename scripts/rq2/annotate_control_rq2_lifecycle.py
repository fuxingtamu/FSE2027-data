"""Re-annotate the frozen control CMOs with the unified RQ2 lifecycle codebook."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SZZ_ROOT = ROOT / "src/output/human_control/szz"
JUDGMENT = ROOT / "src/output/blind_obligation_reclassification/full_v1.jsonl"
PAIRS = ROOT / "src/output/human_control/matched_pairs_all.json"
OUT_ROOT = ROOT / "src/output/human_control/control_rq2_lifecycle"
REPO_AUDIT = ROOT / "src/output/full_scale/rq2_control_repository_evidence.591.json"
sys.path.insert(0, str(ROOT / "src/phase2_szz"))
import td_classifier_obligation as llm  # noqa: E402


SYSTEM = r"""You are annotating the control arm of a software-engineering lifecycle study.
Do not use authorship or the control label as evidence. Use only the origin PR,
later repair evidence, linked issue/CI evidence, and the supplied obligation
description. Return JSON only.

For every supplied accepted obligation, return exactly one object with:
- origin_mechanism: SCOPE_REQUIREMENT_OMISSION, IMPLEMENTATION_SHORTCUT,
  CONTRACT_ASSUMPTION_MISMATCH, STRUCTURAL_CONSTRAINT,
  VERIFICATION_FEEDBACK_GAP, CONTEXT_EVOLUTION, EMERGENT_CONTEXT_CHANGE,
  or UNKNOWN. This explains why the obligation was left unresolved at origin;
  do not copy the TD subtype.
- temporal_status: YES if the obligation was applicable at origin merge,
  NO if it became applicable only after later evolution, UNCERTAIN otherwise.
- exposure_pathway: P-A if failure/user report/issue/CI evidence exposed it;
  P-B only if a later production-code/interface/caller/requirement change
  made the existing obligation observable; P-C if it was found during
  internal maintenance, refactoring, test expansion, or cleanup without a
  concrete failure or later production-context trigger; P-U if evidence is
  insufficient. A repair that merely adds tests or refactors code is P-C, not
  P-B. This is an observation route, not a cause.
- Apply the pathway decision in this order: (1) use P-A only when the supplied
  evidence explicitly names a failure, user report, issue, CI/test failure, or
  static-analysis finding that exposed the obligation; (2) otherwise use P-B
  only when the evidence identifies a later production-code, interface,
  caller, or requirement change that made an already-existing obligation
  observable; (3) use P-C for test expansion, lint/style cleanup, refactoring,
  deletion, or internal maintenance without either kind of trigger; (4) use
  P-U when the evidence cannot distinguish these routes. Do not promote a
  repair to P-A merely because it changes behavior, and do not promote a
  test-only repair to P-B merely because it follows a feature change.
- repair_response: RP1 guard/check/error handling; RP2 logic/contract/type
  correction; RP3 refactoring/deletion; RP4 style/documentation/convention;
  RP5 test/tooling; RP6 dependency/configuration.
- confidence: high, medium, or low.
- evidence: concise file/commit evidence for all four decisions.

Do not infer P-A from a commit title alone. If multiple pathways are plausible,
choose the best-supported one and explain the uncertainty. If the supplied
evidence cannot distinguish origin applicability, use UNCERTAIN."""


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def short(value: object, limit: int = 3500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "\n... [truncated] ..."


def selected() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    valid = {(r.get("cohort"), r.get("pr_key")): r for r in load_jsonl(JUDGMENT) if r.get("valid") is True}
    keys = set()
    for p in pairs:
        if p["ai"].get("merged_at", "")[:10] <= "2026-03-31":
            ak = f"{p['ai']['repo']}#{p['ai']['number']}"
            ck = f"{p['control']['repo']}#{p['control']['number']}"
            if ("AI", ak) in valid and ("CONTROL", ck) in valid:
                keys.add(ck)
    controls = {
        k: valid[("CONTROL", k)] for k in keys
        if valid.get(("CONTROL", k))
        and valid[("CONTROL", k)].get("judgment", {}).get("obligations")
    }
    candidates: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(SZZ_ROOT.glob("*/data/03_szz_enriched.json")):
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = f"{row.get('repo_name')}#{row.get('pr_number')}"
            if key in controls:
                candidates[key].append(row)
    return controls, candidates


def repository_audit_by_case() -> dict[str, dict]:
    """Load repository-level facts without treating cues as labels."""
    if not REPO_AUDIT.exists():
        return {}
    obj = json.loads(REPO_AUDIT.read_text(encoding="utf-8"))
    result = {}
    for row in obj.get("rows", []):
        for candidate in row.get("candidate_audits", []):
            case_id = candidate.get("candidate_case_id")
            if case_id:
                result[case_id] = candidate
    return result


def prompt(key: str, record: dict, candidates: list[dict], repo_audit: dict[str, dict]) -> str:
    parts = [f"PR: {key}", f"Origin title: {record.get('judgment', {}).get('notes', '')[:500]}",
             "\nACCEPTED OBLIGATIONS:",
             "IMPORTANT: output exactly the obligation_id values listed below (for example O1, O2, O3). C1/C2/... are repository candidate labels, not obligation IDs; never output them as obligations."]
    for o in record.get("judgment", {}).get("obligations", []):
        parts += [f"\n{o.get('obligation_id')}: {o.get('description')}",
                  f"legacy TD subtype (do not use as cause): {o.get('subtype')}",
                  f"existing root-cause evidence: {short(o.get('root_cause'))}",
                  f"affected files: {o.get('affected_files')}",
                  f"repayment commits: {o.get('repayment_commits')}",
                  f"obligation evidence: {short(o.get('evidence'), 5000)}"]
    parts.append("\nREPOSITORY CANDIDATES (origin patch and downstream repair):")
    for i, c in enumerate(candidates, 1):
        parts += [f"\nC{i}: {c.get('filepath')} | fix {c.get('commit_date')} | {c.get('commit_message', '')[:220]}",
                  "ORIGIN PATCH:", "```diff", short(c.get('pr_file_patch')), "```",
                  "REPAIR DIFF:", "```diff", short((c.get('fix_diff') or {}).get('diff')), "```",
                  f"linked issues: {short(c.get('linked_issues'), 1200)}",
                  f"issue refs found: {c.get('issue_refs_found')}"]
        audit = repo_audit.get(c.get("case_id"), {})
        if audit:
            parts += [
                "REPOSITORY AUDIT FACTS (facts/cues only; do not treat cues as labels):",
                f"commit_exists={audit.get('commit_exists')}; parent_available={audit.get('parent_available')}; changed_file_count={audit.get('changed_file_count')}",
                f"changed_files={short(audit.get('changed_files'), 2500)}",
                f"origin_file_changed={audit.get('origin_file_changed')}; cross_file_change={audit.get('cross_file_change')}",
                f"touches_test_file={audit.get('touches_test_file')}; touches_documentation={audit.get('touches_documentation')}; touches_configuration={audit.get('touches_configuration')}",
                f"interface_contract_diff_cue={audit.get('interface_contract_diff_cue')}; caller_consumer_diff_cue={audit.get('caller_consumer_diff_cue')}; ci_failure_diff_cue={audit.get('ci_failure_diff_cue')}",
            ]
    parts.append("\nReturn {\"obligations\":[{\"obligation_id\":\"O1\",\"origin_mechanism\":...,\"temporal_status\":...,\"exposure_pathway\":...,\"repair_response\":...,\"confidence\":...,\"evidence\":...}]}.")
    return "\n".join(parts)


def validate(raw: dict, ids: set[str]) -> tuple[bool, list[str]]:
    got = {str(x.get("obligation_id")) for x in raw.get("obligations", [])} if isinstance(raw, dict) else set()
    errors = []
    if got != ids:
        errors.append(f"obligation coverage missing={sorted(ids-got)} extra={sorted(got-ids)}")
    allowed_t = {"YES", "NO", "UNCERTAIN"}
    allowed_p = {"P-A", "P-B", "P-C", "P-U"}
    allowed_m = {"SCOPE_REQUIREMENT_OMISSION", "IMPLEMENTATION_SHORTCUT", "CONTRACT_ASSUMPTION_MISMATCH", "STRUCTURAL_CONSTRAINT", "VERIFICATION_FEEDBACK_GAP", "CONTEXT_EVOLUTION", "EMERGENT_CONTEXT_CHANGE", "UNKNOWN"}
    allowed_r = {f"RP{i}" for i in range(1, 7)}
    for x in raw.get("obligations", []) if isinstance(raw, dict) else []:
        for field, allowed in (("temporal_status", allowed_t), ("exposure_pathway", allowed_p), ("origin_mechanism", allowed_m), ("repair_response", allowed_r)):
            if x.get(field) not in allowed:
                errors.append(f"{x.get('obligation_id')} invalid {field}={x.get(field)}")
    return not errors, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pr", nargs="*", default=[], help="process only these PR keys")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--checkpoint", default=str(OUT_ROOT / "control_rq2_lifecycle.jsonl"))
    ap.add_argument("--output", default=str(OUT_ROOT / "control_rq2_lifecycle.json"))
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    controls, candidates = selected()
    repo_audit = repository_audit_by_case()
    prompts = {k: prompt(k, r, candidates.get(k, []), repo_audit) for k, r in sorted(controls.items())}
    if args.limit:
        prompts = dict(list(prompts.items())[:args.limit])
    if args.pr:
        prompts = {k: prompts[k] for k in args.pr if k in prompts}
    done = {}
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                x = json.loads(line); done[x["pr_key"]] = x
            except (ValueError, KeyError):
                pass
    # Retry invalid checkpoint entries as well; only validated judgments are
    # considered complete. This is important after transient JSON/schema
    # failures in a long batch.
    todo = [(k, controls[k]) for k in prompts if k not in done or done[k].get("valid") is not True]
    print(f"selected control PRs={len(prompts)}; todo={len(todo)}", flush=True)

    def one(key: str, record: dict) -> dict:
        raw = llm.call_llm(SYSTEM, prompts[key], model=args.model, max_retries=6, max_tokens=5000)
        ids = {str(o.get("obligation_id")) for o in record.get("judgment", {}).get("obligations", [])}
        ok, errors = validate(raw, ids) if isinstance(raw, dict) and "error" not in raw else (False, ["LLM error"])
        return {"pr_key": key, "n_obligations": len(ids), "prompt_chars": len(prompts[key]), "judgment": raw, "valid": ok, "validation_errors": errors}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, checkpoint.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(one, k, r) for k, r in todo]
        for i, future in enumerate(as_completed(futures), 1):
            item = future.result(); done[item["pr_key"]] = item
            fh.write(json.dumps(item, ensure_ascii=False) + "\n"); fh.flush()
            print(f"[{i}/{len(todo)}] {item['pr_key']} valid={item['valid']}", flush=True)
    output = Path(args.output)
    output.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pr_judgments": len(done), "valid": sum(x.get('valid') is True for x in done.values()), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
