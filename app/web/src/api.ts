/**
 * The only module that talks to the backend.
 *
 * Same boundary rule as `ragkit/gemini.py` being the only file that touches the
 * Gemini SDK: when a response shape changes, exactly one file needs editing.
 *
 * EVIDENCE KINDS ARE NOT INVENTED HERE. The backend derives them from stored
 * provenance (`text_provenance`, `text_source`, `quote_status`) and sends the
 * label with them. The UI colours by the field it is given -- it does not
 * inspect text and guess. That is what makes the colour trustworthy: amber
 * *means* a model read a chart, rather than looking like it might have.
 */

export type EvidenceKind =
  | "quoted"
  | "structure_inferred"
  | "assistant_reading"
  | "found_not_quoted"
  | "conversation";

export type QuoteStatus = "verified" | "absent" | "unquotable" | "no_quote" | null;

export interface Citation {
  label: number;
  quote: string | null;
  chunk_id: string | null;
  evidence_kind: EvidenceKind | null;
  evidence_label: string;
  quote_status: QuoteStatus;
  fabricated: boolean;
  overlap: number;
  detail: string;
  locator: string | null;
}

export interface Claim {
  text: string;
  source: "documents" | "conversation";
  evidence_kind: EvidenceKind | null;
  citations: Citation[];
}

export interface SourceRef {
  label: number;
  chunk_id: string;
  source_id: string;
  page: number | null;
  page_end: number | null;
  page_label?: string;
  kind: string;
  text_source: string;
  provenance: string;
  heading_path: string[];
  highlightable?: boolean;
  quote?: string | null;
  asset_path?: string | null;
}

export interface Understanding {
  original: string;
  search_query: string;
  route: "documents" | "documents_and_conversation" | "conversation_only" | "ask_fresh";
  was_rewritten: boolean;
  topic_shift: boolean;
  read_as: string;
  suggested_fresh_question: string;
  fallback: boolean;
  needs_retrieval: boolean;
  header: string;
}

export interface AskResponse {
  question: string;
  understanding: Understanding;
  answer_markdown: string;
  abstained: boolean;
  abstain_reason: string;
  suggested_question?: string;
  grounded: boolean;
  reasons: string[];
  claims: Claim[];
  evidence_mix: Record<EvidenceKind, number>;
  reconciliation: Record<string, number | string>;
  citation_integrity: { name: string; rule: string; observed: string; state: string };
  conversation_attribution: string;
  sources: SourceRef[];
  retrieval: {
    mode: string;
    budget: number;
    ran: boolean;
    skipped_because: string | null;
    children_considered: number;
    parents_delivered: number;
    child_tokens: number;
    parent_tokens: number;
    leg_stats: Record<string, any> | null;
    fusion_explain: string | null;
  };
  usage: { prompt_tokens: number; output_tokens: number; cached_tokens: number };
  timings_ms: Record<string, number>;
  conversation?: { id: string; title: string; n_turns: number; drift: Drift };
}

export interface Drift {
  turns: number;
  conversation_only: number;
  conversation_only_share: number | null;
  first_half_share: number | null;
  second_half_share: number | null;
  sufficient: boolean;
  drifting: boolean;
  note: string;
}

export interface SourceDetail {
  chunk_id: string;
  source_id: string;
  locator: string;
  page: number | null;
  page_end: number | null;
  kind: string;
  text_source: string;
  provenance: string;
  evidence_kind: EvidenceKind;
  evidence_label: string;
  note: string;
  heading_path: string[];
  text: string;
  verbatim: string | null;
  highlight: { start: number; end: number } | null;
  asset_url: string | null;
  table_header_missing: boolean;
  table_continuation_suspect: boolean;
}

/** One entry in the corpus listing. Named so the sidebar can fold page renders
 *  under the PDF they were extracted from rather than listing them as peers. */
export interface Doc {
  source_id: string;
  title: string;
  doc_type: string;
  pages: number;
  chunks: number;
  tables: number;
  continuation_suspects: number;
  state: "READY" | "SEARCHABLE_INCOMPLETE";
}

export interface StatusResponse {
  documents: Doc[];
  tombstones: Record<string, any>;
  index: Record<string, any>;
  models: Record<string, string>;
  capabilities: { resolved: Record<string, string>; probes: Record<string, any> };
  degradations: any[];
  scope_note: string;
  // Present when the backend is deployed read-only. Optional so a local backend
  // that predates the field still typechecks rather than needing a lockstep
  // deploy of both halves.
  demo?: {
    read_only: boolean;
    // Distinct from read_only on purpose: the demo refuses deletion and
    // accepts uploads, so one flag cannot drive both controls.
    uploads_enabled?: boolean;
    upload_limits?: string;
    why: string;
    rate_limit: string;
  };
}

export interface Job {
  id: string;
  kind: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: { stage: string; current: number; total: number; detail: string };
  result: any;
  error: string | null;
}

export interface Rate {
  n: number;
  hits: number;
  rate: number | null;
  sufficient: boolean;
  label: string;
  ci95?: [number, number];
}

export interface Check {
  name: string;
  rule: string;
  observed: string;
  state: "HOLDS" | "FAILS" | "NOT_MEASURED";
  n: number | null;
  fingerprint: string[];
  detail: string;
  why: string;
}

export interface InspectorResponse {
  /** Measured failures by Barnett failure point, written by
   *  scripts/failure_histogram.py and read here rather than recomputed. */
  failures?: {
    n: number;
    budget: number;
    thesis?: string;
    points: { id: string; name: string; n: number; share: number; observable: string }[];
    by_cause: Record<string, number>;
    by_stratum: Record<string, number>;
  } | null;
  reconciliation: {
    fingerprint: string[];
    pipeline_fingerprint: string | null;
    trusted: boolean;
    summary: { failing: number; passing: number; not_measured: number };
    thesis: string;
    checks: Check[];
  };
  judge_gate: {
    may_emit_judged_metrics: boolean;
    validated: boolean;
    judge_model: string;
    generator_model: string;
    self_grading_collision: boolean;
    withheld_because: string | null;
    detail: Record<string, any>;
    note: string;
  };
  judged: any;
  deferred: {
    deferrals: {
      name: string;
      guide_module: string;
      decision: string;
      because: string;
      revisit_when: string;
      cost_if_wrong: string;
      expired: boolean;
      evidence: string;
    }[];
    expired: string[];
    note: string;
  };
  eval: {
    token_budget: number;
    headline: Record<string, Rate | number>;
    scope_label: string;
    stratum_coverage: {
      declared: string[];
      present: string[];
      missing: string[];
      thin: string[];
    };
    by_stratum: Record<string, Record<string, Rate | number>>;
    by_anchor: Record<string, Record<string, Rate | number>>;
    budget_sweep: Record<string, Record<string, Rate | number>>;
    regression_tests: Record<
      string,
      { passes_at_budget: number | null; passes_at_headline: boolean; needles: number }
    >;
    golden_set: Record<string, any>;
    index_provenance: Record<string, any>;
    seconds: number;
  } | null;
  baseline: { headline: Record<string, Rate>; index_provenance: Record<string, any> } | null;
  hybrid_comparison: {
    budgets: number[];
    modes: string[];
    bm25: Record<string, any>;
    rrf_k: number;
    results: Record<string, any>;
  } | null;
  goldenset_report: any;
}

export interface Conversation {
  id: string;
  title: string;
  n_turns: number;
  updated_at: string;
  conversation_only_share: number | null;
  drifting: boolean;
}

// --------------------------------------------------------------------------

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    // Surface the server's own message. A generic "request failed" would hide
    // the 410-with-tombstone case, which the UI needs to render differently
    // from a 404 -- "this was removed" and "this never existed" are different
    // answers and collapsing them makes a correct old answer look fabricated.
    let detail: unknown = null;
    try {
      detail = (await res.json())?.detail ?? null;
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(
      typeof detail === "string" ? detail : `${res.status} ${res.statusText}`,
    ) as Error & { status?: number; detail?: unknown };
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return (await res.json()) as T;
}

export const api = {
  status: () => req<StatusResponse>("/api/status"),
  inspector: () => req<InspectorResponse>("/api/inspector"),

  ask: (body: {
    question: string;
    budget?: number | null;
    sources?: number;
    mode?: "dense" | "sparse" | "rrf";
    history?: string;
  }) => req<AskResponse>("/api/ask", { method: "POST", body: JSON.stringify(body) }),

  source: (chunkId: string, quote?: string | null) =>
    req<SourceDetail>(
      `/api/source/${encodeURIComponent(chunkId)}` +
        (quote ? `?quote=${encodeURIComponent(quote)}` : ""),
    ),

  conversations: () => req<{ conversations: Conversation[] }>("/api/conversations"),
  createConversation: (title = "") =>
    req<{ id: string }>(`/api/conversations?title=${encodeURIComponent(title)}`, {
      method: "POST",
    }),
  conversation: (id: string) => req<any>(`/api/conversations/${id}`),
  deleteConversation: (id: string) =>
    req<{ deleted: boolean }>(`/api/conversations/${id}`, { method: "DELETE" }),
  askIn: (
    id: string,
    body: { question: string; budget?: number | null; sources?: number; mode?: string },
  ) =>
    req<AskResponse>(`/api/conversations/${id}/ask`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  upload: async (files: File[], thorough = true) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const res = await fetch(`/api/documents?thorough=${thorough ? 1 : 0}`, {
      method: "POST",
      body: fd,
    });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
    return (await res.json()) as {
      saved: { name: string; bytes: number; source_id: string }[];
      rejected: { name: string; reason: string }[];
      // Present whenever anything was accepted; the indexing runs in the
      // background because it cannot finish inside the request.
      job?: Job | null;
      mode?: "thorough" | "fast";
      next: string;
    };
  },
  ingest: () => req<{ job: Job; queued_behind: number; note: string }>("/api/ingest", {
    method: "POST",
  }),
  removeDocument: (sourceId: string) =>
    req<{ job: Job; will_remove: Record<string, number>; note: string }>(
      `/api/documents/${encodeURIComponent(sourceId)}`,
      { method: "DELETE" },
    ),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  jobs: () => req<{ active: Job | null; queued: number; jobs: Job[] }>("/api/jobs"),

  flag: (body: {
    verdict: "not_in_source" | "source_doesnt_say" | "source_is_fine" | "helpful";
    conversation_id?: string;
    turn_index?: number;
    claim_index?: number;
    claim_text?: string;
    chunk_id?: string;
    note?: string;
  }) => req<{ recorded: any; label: string }>("/api/feedback", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  feedbackStats: () => req<any>("/api/feedback"),
};

// --------------------------------------------------------------------------
// Evidence presentation -- one table, so the colour and the words cannot drift
// apart across components.

export const EVIDENCE: Record<
  EvidenceKind,
  { chip: string; dot: string; short: string; long: string }
> = {
  quoted: {
    chip: "text-quote-600 border-quote-600/30 bg-quote-600/5",
    dot: "bg-quote-600",
    short: "quoted",
    long: "Quoted exactly from your documents",
  },
  structure_inferred: {
    chip: "text-ink-600 border-ink-400/40 bg-paper-300/50",
    dot: "bg-ink-500",
    short: "structure inferred",
    long: "Document text — the table's row/column structure was lost in extraction and inferred",
  },
  assistant_reading: {
    chip: "text-chart-600 border-chart-600/30 bg-chart-600/5",
    dot: "bg-chart-600",
    short: "read from a chart",
    long: "The assistant's reading of an image — not a quotation",
  },
  found_not_quoted: {
    chip: "text-ink-500 border-ink-400/40 bg-paper-200",
    dot: "bg-ink-400",
    short: "found, not quoted",
    long: "It's in your documents. It is not being quoted, because the quote did not verify.",
  },
  conversation: {
    chip: "text-ink-500 border-ink-400/30 bg-paper-200",
    dot: "bg-ink-400",
    short: "this conversation",
    long: "Said earlier in this conversation — not from your documents",
  },
};
