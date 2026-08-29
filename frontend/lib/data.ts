import "server-only";

import fs from "node:fs";
import path from "node:path";

import type {
  BaselineCostLatency,
  HoldoutConsumed,
  HoldoutReport,
  RetrievalEval,
  RouterModel,
  SampleReport,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "..", "data", "processed");

/**
 * READ-ONLY access to this project's real, already-measured result
 * artifacts (`../results/*.json`, produced by the Python evaluation
 * pipeline — never by this frontend). Nothing in this module writes to
 * disk, calls a network API, or constructs an LLM client — it only parses
 * JSON that already exists on disk at build time. There is no code path
 * here that can reach `data/processed/final_holdout.json` itself (that
 * file was consumed once, by the evaluation scripts, months before this
 * frontend existed) — only the aggregated *reports* derived from it.
 *
 * `RESULTS_DIR` points one level up from the Next.js app root, at the
 * same `results/` directory the rest of this repository already uses as
 * its single source of truth — intentionally not copied or duplicated
 * into `frontend/`, so there is exactly one place these numbers can live.
 */
const RESULTS_DIR = path.join(process.cwd(), "..", "results");

function readJSON<T>(filename: string): T {
  const filePath = path.join(RESULTS_DIR, filename);
  const raw = fs.readFileSync(filePath, "utf-8");
  return JSON.parse(raw) as T;
}

export function getSampleReport(): SampleReport {
  return readJSON<SampleReport>("phase9_sample_report.json");
}

export function getHoldoutReport(): HoldoutReport {
  return readJSON<HoldoutReport>("phase9_holdout_report.json");
}

export function getRetrievalEval(): RetrievalEval {
  return readJSON<RetrievalEval>("retrieval_eval_development.json");
}

export function getRouterModel(): RouterModel {
  return readJSON<RouterModel>("learned_router_model.json");
}

export function getHoldoutConsumed(): HoldoutConsumed {
  return readJSON<HoldoutConsumed>("final_holdout_consumed.json");
}

/**
 * Dense/Hybrid/Hybrid+Reranker don't get their own aggregated cost/latency
 * fields in phase9_sample_report.json (only Always-Agentic/Adaptive do,
 * since those are this project's primary comparison) — so, exactly like
 * `scripts/generate_phase9_charts.py` did for the README charts, this
 * derives their mean cost/query and latency/query directly from the raw
 * per-question traces, filtered to the same frozen 50-question sample.
 * Every number here is a plain arithmetic mean over real recorded fields
 * — nothing invented, nothing hardcoded.
 */
export function getDevBaselineCostLatency(): Record<
  "dense" | "hybrid" | "hybrid_reranker",
  BaselineCostLatency
> {
  const sample = readJSON<{ qa_ids: string[] }>("phase9_sample.json");
  const sampleIds = new Set(sample.qa_ids);

  const pipelines = ["dense", "hybrid", "hybrid_reranker"] as const;
  const result = {} as Record<(typeof pipelines)[number], BaselineCostLatency>;

  for (const pipeline of pipelines) {
    const raw = readJSON<{
      records: Array<{ qa_id: string; total_cost_usd: number; total_latency_ms: number }>;
    }>(`phase9_${pipeline}_raw.json`);
    const records = raw.records.filter((r) => sampleIds.has(r.qa_id));
    const n = records.length;
    result[pipeline] = {
      cost: records.reduce((sum, r) => sum + r.total_cost_usd, 0) / n,
      latency: records.reduce((sum, r) => sum + r.total_latency_ms, 0) / n,
    };
  }

  return result;
}

/**
 * A handful of REAL questions from data/processed/dev_subset.json — the
 * development split, never final_holdout — for the live demo's example
 * chips. Selected deterministically (fixed indices into the non-null,
 * question-type-sorted list, capped to a readable length) so the same
 * five questions appear on every build; nothing here is written by hand.
 */
export function getExampleQuestions(): string[] {
  const filePath = path.join(DATA_DIR, "dev_subset.json");
  const raw = fs.readFileSync(filePath, "utf-8");
  const records = JSON.parse(raw) as Array<{ query: string; question_type: string }>;

  const byType = new Map<string, string[]>();
  for (const r of records) {
    if (r.question_type === "null_query" || r.query.length > 220) continue;
    const bucket = byType.get(r.question_type) ?? [];
    bucket.push(r.query);
    byType.set(r.question_type, bucket);
  }

  const examples: string[] = [];
  for (const type of ["inference_query", "comparison_query", "temporal_query"]) {
    const bucket = byType.get(type);
    if (!bucket || bucket.length === 0) continue;
    // Shortest-first within the length-filtered bucket — a scannable example
    // chip, not necessarily the first one encountered in the source file.
    const shortest = [...bucket].sort((a, b) => a.length - b.length)[0];
    examples.push(shortest);
  }
  return examples;
}
