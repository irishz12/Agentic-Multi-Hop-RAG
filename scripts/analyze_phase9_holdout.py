#!/usr/bin/env python
"""FINAL HOLDOUT evaluation — OFFLINE aggregation. No live calls, no
Mantle client, and (critically) NO further access to
data/processed/final_holdout.json — gold document ids for evidence-
coverage scoring come from `results/phase9_holdout_sample.json`'s
persisted `gold_doc_ids_by_qa_id` field (written once, at selection time),
not a second file read.

INTEGRITY CHECK (honors the promise made in
scripts/freeze_final_evaluation_manifest.py's docstring): re-hashes the
exact same file list recorded in results/final_evaluation_manifest.json
and RAISES if a single hash differs from the pre-access snapshot — this is
the mechanical proof that nothing was tuned, retrained, or edited between
freezing the manifest and finishing the holdout evaluation, not just a
claim.

Combines:
  - results/final_evaluation_manifest.json (pre-access hashes, re-verified)
  - results/phase9_holdout_sample.json (the frozen 50-qa_id holdout
    selection + gold_doc_ids_by_qa_id)
  - results/phase9_holdout_{always_agentic,adaptive}_raw.json
  - results/phase9_holdout_judge_{always_agentic,adaptive}.json
  - results/phase9_sample_report.json (the DEVELOPMENT-sample report, for
    the development-vs-holdout comparison)

Writes results/phase9_holdout_report.json.

Usage:
    python scripts/analyze_phase9_holdout.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from mhrag.config import PROJECT_ROOT
from mhrag.eval.answer_metrics import exact_match, is_abstention, token_f1

PIPELINES = ("always_agentic", "adaptive")


def _verify_manifest_unchanged() -> dict:
    manifest_path = PROJECT_ROOT / "results" / "final_evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mismatches = []
    for rel_path, frozen_hash in manifest["file_hashes_sha1"].items():
        current_hash = hashlib.sha1((PROJECT_ROOT / rel_path).read_bytes()).hexdigest()
        if current_hash != frozen_hash:
            mismatches.append(rel_path)
    if mismatches:
        raise SystemExit(
            f"INTEGRITY VIOLATION — {len(mismatches)} frozen file(s) changed since the pre-access "
            f"manifest was written, which is not permitted for a one-time holdout evaluation: {mismatches}"
        )
    print(f"Integrity check PASSED — all {len(manifest['file_hashes_sha1'])} frozen files unchanged "
          f"since manifest (generated_at={manifest['generated_at']})")
    return manifest


def _load_sample_records(pipeline: str) -> dict[str, dict]:
    raw = json.loads((PROJECT_ROOT / "results" / f"phase9_holdout_{pipeline}_raw.json").read_text())
    return {r["qa_id"]: r for r in raw["records"]}


def _load_judge_scores(pipeline: str) -> dict[str, dict]:
    d = json.loads((PROJECT_ROOT / "results" / f"phase9_holdout_judge_{pipeline}.json").read_text())
    return {r["qa_id"]: r for r in d["records"]}


def main() -> None:
    manifest = _verify_manifest_unchanged()

    sample = json.loads((PROJECT_ROOT / "results" / "phase9_holdout_sample.json").read_text())
    sample_ids = set(sample["qa_ids"])
    gold_doc_ids_by_qa_id = {qid: set(docs) for qid, docs in sample["gold_doc_ids_by_qa_id"].items()}
    print(f"Holdout sample: {len(sample_ids)} qa_ids (seed={sample['seed']})")

    pipeline_records = {p: _load_sample_records(p) for p in PIPELINES}
    for p in PIPELINES:
        assert set(pipeline_records[p]) == sample_ids, f"{p} raw checkpoint doesn't exactly match the 50-qa_id sample"

    judge_scores = {p: _load_judge_scores(p) for p in PIPELINES}

    null_ids = {qid for qid, r in pipeline_records["always_agentic"].items() if r["question_type"] == "null_query"}
    non_null_ids = sample_ids - null_ids
    print(f"non-null: {len(non_null_ids)}, null: {len(null_ids)}")

    # --- item 1: distribution (already in phase9_holdout_sample.json, echoed here) --------
    distribution = {
        "question_type_distribution": sample["question_type_distribution"],
        "stratum_distribution": sample["stratum_distribution"],
    }

    # --- deterministic metrics (both pipelines) --------------------------------------------
    deterministic_metrics: dict[str, dict] = {}
    for p in PIPELINES:
        recs = pipeline_records[p]
        em_vals = [exact_match(recs[qid]["answer"], recs[qid]["gold_answer"]) for qid in non_null_ids]
        f1_vals = [token_f1(recs[qid]["answer"], recs[qid]["gold_answer"]) for qid in non_null_ids]
        abstain_correct = [int(is_abstention(recs[qid]["answer"])) for qid in null_ids]
        deterministic_metrics[p] = {
            "normalized_exact_match": sum(em_vals) / len(em_vals),
            "token_f1": sum(f1_vals) / len(f1_vals),
            "null_query_abstention_accuracy": sum(abstain_correct) / len(abstain_correct),
        }

    # --- item 2: judge scores (both, with and without fallback sensitivity) ----------------
    judge_summary: dict[str, dict] = {}
    for p in PIPELINES:
        all_scores = [judge_scores[p][qid]["score"] for qid in non_null_ids]
        non_fallback_scores = [judge_scores[p][qid]["score"] for qid in non_null_ids
                                if not judge_scores[p][qid]["fallback_used"]]
        n_fallback = len(all_scores) - len(non_fallback_scores)
        judge_summary[p] = {
            "mean_judge_score_including_fallbacks": sum(all_scores) / len(all_scores),
            "mean_judge_score_excluding_fallbacks": (
                sum(non_fallback_scores) / len(non_fallback_scores) if non_fallback_scores else None
            ),
            "n_correct": sum(1 for s in all_scores if s == 1.0),
            "n_partially_correct": sum(1 for s in all_scores if s == 0.5),
            "n_incorrect": sum(1 for s in all_scores if s == 0.0),
            "n_judge_fallbacks": n_fallback,
            "fallback_qa_ids": [qid for qid in non_null_ids if judge_scores[p][qid]["fallback_used"]],
        }

    def combined_quality(pipeline: str, qid: str) -> float:
        if qid in null_ids:
            return float(is_abstention(pipeline_records[pipeline][qid]["answer"]))
        return judge_scores[pipeline][qid]["score"]

    combined_quality_mean = {p: sum(combined_quality(p, qid) for qid in sample_ids) / 50 for p in PIPELINES}

    # --- item 3: quality retention ----------------------------------------------------------
    adaptive_quality = combined_quality_mean["adaptive"]
    agentic_quality = combined_quality_mean["always_agentic"]
    quality_retention_pct = (adaptive_quality / agentic_quality) if agentic_quality > 0 else None

    # --- item 5: evidence coverage -----------------------------------------------------------
    def evidence_coverage(pipeline: str, qid: str):
        gold = gold_doc_ids_by_qa_id[qid]
        if not gold:
            return None
        used = set(pipeline_records[pipeline][qid]["evidence_doc_ids_used"])
        return len(gold & used) / len(gold)

    evidence_coverage_mean = {
        p: sum(v for v in (evidence_coverage(p, qid) for qid in non_null_ids)) / len(non_null_ids)
        for p in PIPELINES
    }

    # --- item 6/7: cost/latency + reduction ---------------------------------------------------
    ag_costs = [pipeline_records["always_agentic"][qid]["total_cost_usd"] for qid in sample_ids]
    ad_costs = [pipeline_records["adaptive"][qid]["total_cost_usd"] for qid in sample_ids]
    ag_lat = [pipeline_records["always_agentic"][qid]["total_latency_ms"] for qid in sample_ids]
    ad_lat = [pipeline_records["adaptive"][qid]["total_latency_ms"] for qid in sample_ids]
    mean_ag_cost, mean_ad_cost = sum(ag_costs) / 50, sum(ad_costs) / 50
    mean_ag_lat, mean_ad_lat = sum(ag_lat) / 50, sum(ad_lat) / 50
    cost_reduction_pct = (mean_ag_cost - mean_ad_cost) / mean_ag_cost
    latency_reduction_pct = (mean_ag_lat - mean_ad_lat) / mean_ag_lat

    # --- item 8/9: breakdown by query type and hop count --------------------------------------
    # question_type/hop_count are read straight off the already-completed raw checkpoints
    # (persisted there at benchmark time) — no need to touch final_holdout.json again.
    def group_breakdown(group_fn) -> dict:
        groups: dict[str, list[str]] = {}
        for qid in sample_ids:
            rec = pipeline_records["always_agentic"][qid]
            groups.setdefault(group_fn(rec), []).append(qid)
        breakdown = {}
        for key, qids in groups.items():
            row = {"n": len(qids)}
            for p in PIPELINES:
                vals = [combined_quality(p, qid) for qid in qids]
                row[f"{p}_mean_quality"] = sum(vals) / len(vals)
            breakdown[key] = row
        return breakdown

    breakdown_by_question_type = group_breakdown(lambda r: r["question_type"])
    breakdown_by_hop_count = group_breakdown(
        lambda r: f"hop{r['hop_count']}" if r["question_type"] != "null_query" else "null"
    )

    # --- item 10: under-routed Adaptive failures ----------------------------------------------
    under_routed_failures = []
    for qid in non_null_ids:
        route = pipeline_records["adaptive"][qid]["predicted_route"]
        if route not in ("SIMPLE", "MEDIUM"):
            continue
        ad_score = judge_scores["adaptive"][qid]["score"]
        ag_score = judge_scores["always_agentic"][qid]["score"]
        ad_cov = evidence_coverage("adaptive", qid)
        ag_cov = evidence_coverage("always_agentic", qid)
        if ad_score < ag_score or (ad_cov is not None and ag_cov is not None and ad_cov < ag_cov):
            rec = pipeline_records["adaptive"][qid]
            under_routed_failures.append(
                {
                    "qa_id": qid, "question_type": rec["question_type"], "hop_count": rec["hop_count"],
                    "route": route, "adaptive_judge_score": ad_score, "always_agentic_judge_score": ag_score,
                    "adaptive_evidence_coverage": ad_cov, "always_agentic_evidence_coverage": ag_cov,
                }
            )

    # --- item 11: development vs holdout comparison -------------------------------------------
    dev_report_path = PROJECT_ROOT / "results" / "phase9_sample_report.json"
    dev_vs_holdout = None
    if dev_report_path.exists():
        dev_report = json.loads(dev_report_path.read_text())
        dev_vs_holdout = {
            "development": {
                "combined_quality_mean": dev_report["combined_quality_mean"],
                "adaptive_quality_retention_pct": dev_report["adaptive_quality_retention_pct_vs_always_agentic"],
                "cost_reduction_pct": dev_report["cost_latency"]["cost_reduction_pct"],
                "latency_reduction_pct": dev_report["cost_latency"]["latency_reduction_pct"],
                "evidence_coverage_mean": {
                    k: v for k, v in dev_report["evidence_coverage_mean"].items() if k in PIPELINES
                },
            },
            "holdout": {
                "combined_quality_mean": combined_quality_mean,
                "adaptive_quality_retention_pct": quality_retention_pct,
                "cost_reduction_pct": cost_reduction_pct,
                "latency_reduction_pct": latency_reduction_pct,
                "evidence_coverage_mean": evidence_coverage_mean,
            },
        }

    # --- item 12: judge fallback sensitivity (already computed per-pipeline above) -----------
    total_fallbacks = sum(judge_summary[p]["n_judge_fallbacks"] for p in PIPELINES)

    # --- item 13: total evaluation cost --------------------------------------------------------
    total_pipeline_cost = sum(ag_costs) + sum(ad_costs)
    all_judge_records = [r for p in PIPELINES for r in judge_scores[p].values()]
    total_judge_cost = None  # pricing unverified — never guessed (same rule as the dev-sample report)
    total_evaluation_cost = {
        "total_pipeline_cost_usd": total_pipeline_cost,
        "total_judge_cost_usd": total_judge_cost,
        "n_judge_calls": len(all_judge_records),
        "judge_total_input_tokens": sum(r["input_tokens"] or 0 for r in all_judge_records),
        "judge_total_output_tokens": sum(r["output_tokens"] or 0 for r in all_judge_records),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "FINAL HOLDOUT evaluation — Adaptive vs Always-Agentic, one-time report",
        "split": "final_holdout",
        "sample_seed": sample["seed"],
        "sample_size": 50,
        "n_non_null": len(non_null_ids),
        "n_null": len(null_ids),
        "integrity_check": "PASSED — all frozen files unchanged since pre-access manifest",
        "pre_access_manifest_generated_at": manifest["generated_at"],
        "distribution": distribution,
        "deterministic_metrics": deterministic_metrics,
        "judge_scores": judge_summary,
        "combined_quality_mean": combined_quality_mean,
        "adaptive_quality_retention_pct_vs_always_agentic": quality_retention_pct,
        "evidence_coverage_mean": evidence_coverage_mean,
        "cost_latency": {
            "always_agentic_mean_cost_usd": mean_ag_cost, "adaptive_mean_cost_usd": mean_ad_cost,
            "always_agentic_mean_latency_ms": mean_ag_lat, "adaptive_mean_latency_ms": mean_ad_lat,
            "cost_reduction_pct": cost_reduction_pct, "latency_reduction_pct": latency_reduction_pct,
        },
        "breakdown_by_question_type": breakdown_by_question_type,
        "breakdown_by_hop_count": breakdown_by_hop_count,
        "under_routed_failures": under_routed_failures,
        "development_vs_holdout": dev_vs_holdout,
        "judge_fallback_total": total_fallbacks,
        "total_evaluation_cost": total_evaluation_cost,
    }
    out_path = PROJECT_ROOT / "results" / "phase9_holdout_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
