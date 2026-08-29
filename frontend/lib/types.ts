// Types mirror the exact shape of the read-only JSON artifacts under
// ../results/*.json — see lib/data.ts. Field names match the source files
// verbatim so a diff between this file and the JSON is a fast way to spot
// drift if a future analysis script changes an artifact's shape.

export type PipelineKey = "dense" | "hybrid" | "hybrid_reranker" | "always_agentic" | "adaptive";

export interface DeterministicMetrics {
  normalized_exact_match: number;
  token_f1: number;
  null_query_abstention_accuracy: number;
  n_non_null?: number;
  n_null?: number;
}

export interface SampleJudgeScore {
  mean_judge_score: number;
  n_correct: number;
  n_partially_correct: number;
  n_incorrect: number;
  n_judge_fallbacks: number;
}

export interface HoldoutJudgeScore {
  mean_judge_score_including_fallbacks: number;
  mean_judge_score_excluding_fallbacks: number | null;
  n_correct: number;
  n_partially_correct: number;
  n_incorrect: number;
  n_judge_fallbacks: number;
  fallback_qa_ids: string[];
}

export interface CostLatency {
  always_agentic_mean_cost_usd: number;
  adaptive_mean_cost_usd: number;
  always_agentic_mean_latency_ms: number;
  adaptive_mean_latency_ms: number;
  cost_reduction_pct: number;
  latency_reduction_pct: number;
}

export interface BreakdownRow {
  n: number;
  hybrid_reranker_mean_quality?: number;
  always_agentic_mean_quality: number;
  adaptive_mean_quality: number;
}

export interface UnderRoutedFailure {
  qa_id: string;
  question_type: string;
  hop_count: number;
  route: string;
  adaptive_judge_score: number;
  always_agentic_judge_score: number;
  adaptive_evidence_coverage: number | null;
  always_agentic_evidence_coverage: number | null;
}

export interface SampleReport {
  generated_at: string;
  sample_seed: number;
  sample_size: number;
  n_non_null: number;
  n_null: number;
  deterministic_metrics: Partial<Record<PipelineKey, DeterministicMetrics>>;
  judge_scores: Partial<Record<PipelineKey, SampleJudgeScore>>;
  combined_quality_mean: Partial<Record<PipelineKey, number>>;
  adaptive_quality_retention_pct_vs_always_agentic: number;
  evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
  cost_latency: CostLatency;
  breakdown_by_question_type: Record<string, BreakdownRow>;
  breakdown_by_hop_count: Record<string, BreakdownRow>;
  under_routed_failures: UnderRoutedFailure[];
  judge_call_stats: {
    n_judge_calls: number;
    total_input_tokens: number;
    total_output_tokens: number;
    mean_latency_ms: number;
    n_fallbacks: number;
    total_cost_usd: number | null;
  };
}

export interface HoldoutReport {
  generated_at: string;
  sample_seed: number;
  sample_size: number;
  n_non_null: number;
  n_null: number;
  integrity_check: string;
  pre_access_manifest_generated_at: string;
  deterministic_metrics: Partial<Record<PipelineKey, DeterministicMetrics>>;
  judge_scores: Partial<Record<PipelineKey, HoldoutJudgeScore>>;
  combined_quality_mean: Partial<Record<PipelineKey, number>>;
  adaptive_quality_retention_pct_vs_always_agentic: number;
  evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
  cost_latency: CostLatency;
  breakdown_by_question_type: Record<string, BreakdownRow>;
  breakdown_by_hop_count: Record<string, BreakdownRow>;
  under_routed_failures: UnderRoutedFailure[];
  development_vs_holdout: {
    development: {
      combined_quality_mean: Partial<Record<PipelineKey, number>>;
      adaptive_quality_retention_pct: number;
      cost_reduction_pct: number;
      latency_reduction_pct: number;
      evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
    };
    holdout: {
      combined_quality_mean: Partial<Record<PipelineKey, number>>;
      adaptive_quality_retention_pct: number;
      cost_reduction_pct: number;
      latency_reduction_pct: number;
      evidence_coverage_mean: Partial<Record<PipelineKey, number>>;
    };
  };
  judge_fallback_total: number;
  total_evaluation_cost: {
    total_pipeline_cost_usd: number;
    total_judge_cost_usd: number | null;
    n_judge_calls: number;
    judge_total_input_tokens: number;
    judge_total_output_tokens: number;
  };
}

export interface RetrievalMethodAggregate {
  "recall@4": number;
  "recall@5": number;
  "recall@10": number;
  "hit@4": number;
  "hit@10": number;
  "complete_evidence@4": number;
  "complete_evidence@10": number;
  "mrr@10": number;
  "ndcg@10": number;
}

export interface RetrievalEval {
  counts: {
    total_development_questions: number;
    non_null_evaluated: number;
    null_query_excluded: number;
  };
  aggregate: Record<"dense" | "bm25" | "hybrid" | "hybrid_reranker", RetrievalMethodAggregate>;
}

export interface RouterModelSide {
  feature_names: string[];
  threshold: number;
  trained_on_n_questions: number;
}

export interface RouterModel {
  cv_seed: number;
  n_splits: number;
  stage1: RouterModelSide;
  stage2: RouterModelSide;
}

export interface HoldoutConsumed {
  status: string;
  sample_seed: number;
  sample_size: number;
  pre_access_manifest_generated_at: string;
  holdout_report_generated_at: string;
  integrity_check: string;
}

export interface BaselineCostLatency {
  cost: number;
  latency: number;
}
