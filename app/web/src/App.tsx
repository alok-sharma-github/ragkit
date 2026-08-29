/**
 * The shell: document sidebar, conversation list, composer, and the two screens.
 *
 * WHAT IS DELIBERATELY ABSENT: no scope selector, and no PREPARING / QUEUED
 * page-counter on documents. Both appear in the design, and neither is supported
 * by the backend -- scope filtering does not exist in the retriever, and the
 * parser emits nothing until a document finishes, so a page counter would be
 * invented. Drawing a control that does nothing, or a progress bar over an opaque
 * operation, is the same class of claim as a citation the system cannot verify.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type AskResponse,
  type Conversation,
  type Job,
  type StatusResponse,
} from "./api";
import { AnswerView } from "./components/Answer";
import { Inspector } from "./components/Inspector";
import { DegradationBanner, Spinner } from "./components/primitives";

type Screen = "answers" | "inspector";

/* ------------------------------------------------------------------- sidebar */

function DocumentList({
  status,
  onRemove,
  readOnly,
}: {
  status: StatusResponse | null;
  onRemove: (id: string) => void;
  // Same rule as the uploader: a control that 403s is worse than no control.
  // Delete is the most destructive endpoint in the API (it purges the embedding
  // cache by default), so on a public deployment it is not rendered at all.
  readOnly: boolean;
}) {
  if (!status) return <Spinner label="loading corpus" />;
  const byType = status.documents.reduce<Record<string, number>>((a, d) => {
    a[d.doc_type] = (a[d.doc_type] ?? 0) + 1;
    return a;
  }, {});
  return (
    <div>
      <div className="text-[11px] text-ink-400">
        {status.documents.length} documents ·{" "}
        {Object.entries(byType)
          .map(([k, v]) => `${v} ${k}`)
          .join(", ")}
      </div>
      <ul className="mt-2 space-y-1.5">
        {status.documents.map((d) => (
          <li key={d.source_id} className="group">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-[12px] text-ink-900">{d.source_id}</div>
                <div className="text-[11px] text-ink-400 tabular-nums">
                  {d.pages ? `${d.pages} pages · ` : ""}
                  {d.chunks} passages
                  {d.tables ? ` · ${d.tables} tables` : ""}
                  {d.continuation_suspects
                    ? ` · ${d.continuation_suspects} table may span pages`
                    : ""}
                </div>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                {d.state === "SEARCHABLE_INCOMPLETE" && (
                  <span
                    className="rounded bg-chart-600/10 px-1 text-[10px] uppercase tracking-wide text-chart-600"
                    title="some passages were indexed without contextual prefixes"
                  >
                    incomplete
                  </span>
                )}
                {!readOnly && (
                  <button
                    onClick={() => onRemove(d.source_id)}
                    className="rounded px-1 text-[10px] text-ink-400 opacity-0 transition group-hover:opacity-100 hover:bg-paper-200 hover:text-red-800"
                  >
                    remove
                  </button>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
      {Object.keys(status.tombstones ?? {}).length > 0 && (
        <div className="mt-3 border-t border-paper-300 pt-2">
          <div className="text-[10px] uppercase tracking-wide text-ink-400">removed</div>
          <ul className="mt-1 space-y-0.5">
            {Object.entries(status.tombstones).map(([sid, t]: [string, any]) => (
              <li key={sid} className="text-[11px] text-ink-400">
                {sid}
                {t?.n_chunks != null && ` · ${t.n_chunks} passages`}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function Uploader({
  onDone,
  readOnly,
  readOnlyWhy,
}: {
  onDone: () => void;
  readOnly: boolean;
  readOnlyWhy: string;
}) {
  const [job, setJob] = useState<Job | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const poll = useCallback((id: string) => {
    const tick = async () => {
      const j = await api.job(id);
      setJob(j);
      if (j.state === "queued" || j.state === "running") setTimeout(tick, 1500);
      else {
        onDone();
        if (j.state === "failed") setMsg(j.error?.split("\n")[0] ?? "ingest failed");
      }
    };
    tick();
  }, [onDone]);

  const send = async (files: File[]) => {
    setMsg(null);
    const up = await api.upload(files);
    if (up.rejected.length) {
      setMsg(up.rejected.map((r) => `${r.name}: ${r.reason}`).join(" · "));
    }
    if (up.saved.length) {
      const { job } = await api.ingest();
      setJob(job);
      poll(job.id);
    }
  };

  const running = job && (job.state === "queued" || job.state === "running");

  // READ-ONLY DEPLOYMENT: replace the control, do not disable it.
  //
  // A dropzone that accepts a file and then 403s is a worse experience than no
  // dropzone at all -- the user has already committed an action before learning
  // it was never available. Stating the reason in its place also answers the
  // question the absence would otherwise raise ("is this half-built?").
  if (readOnly) {
    return (
      <div className="rounded-md border border-dashed border-paper-500 bg-paper-100 px-3 py-4">
        <p className="text-[12px] font-medium text-ink-600">Upload is disabled here</p>
        <p className="mt-1 text-[10px] leading-snug text-ink-400">
          {readOnlyWhy ||
            "this deployment shares one free-tier Gemini key, so upload, re-ingest and delete are disabled"}
          . The indexed corpus below is fully queryable. To use the full ingest
          pipeline, clone the repo and run it against your own key.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          send(Array.from(e.dataTransfer.files));
        }}
        className="rounded-md border border-dashed border-paper-500 bg-paper-100 px-3 py-4 text-center"
      >
        <p className="text-[12px] text-ink-500">
          Drop files here, or{" "}
          <button
            onClick={() => inputRef.current?.click()}
            className="text-quote-600 underline"
          >
            browse
          </button>
        </p>
        <p className="mt-1 text-[10px] leading-snug text-ink-400">
          PDF, Word, images. A large PDF takes a couple of minutes and blocks the
          queue — progress is per document, because the parser emits nothing until
          a document finishes.
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          accept=".pdf,.docx,.png,.jpg,.jpeg,.webp"
          onChange={(e) => e.target.files && send(Array.from(e.target.files))}
        />
      </div>

      {running && (
        <div className="mt-2 rounded border border-paper-400 bg-paper-50 px-2.5 py-2">
          <div className="flex items-center justify-between text-[11px] text-ink-600">
            <span>
              {job!.state === "queued" ? "queued" : job!.progress.stage || "working"}
            </span>
            <span className="tabular-nums text-ink-400">
              {job!.progress.total
                ? `${job!.progress.current}/${job!.progress.total}`
                : ""}
            </span>
          </div>
          {job!.progress.detail && (
            <div className="truncate text-[11px] text-ink-400">{job!.progress.detail}</div>
          )}
        </div>
      )}
      {job?.state === "done" && job.result && (
        <div className="mt-2 text-[11px] text-ink-500">
          indexed {job.result.children} passages from {job.result.sources} sources
          {job.result.provenance_ok === false && (
            <span className="ml-1 text-red-800">· provenance check FAILED</span>
          )}
        </div>
      )}
      {msg && <div className="mt-2 text-[11px] text-chart-600">{msg}</div>}
    </div>
  );
}

function ConversationList({
  convs,
  activeId,
  onPick,
  onNew,
}: {
  convs: Conversation[];
  activeId: string | null;
  onPick: (id: string) => void;
  onNew: () => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-500">
          Conversations
        </div>
        <button onClick={onNew} className="text-[11px] text-quote-600 hover:underline">
          new
        </button>
      </div>
      {convs.length === 0 && (
        <p className="mt-1.5 text-[11px] text-ink-400">
          none yet — your questions will appear here
        </p>
      )}
      <ul className="mt-1.5 space-y-1">
        {convs.map((c) => (
          <li key={c.id}>
            <button
              onClick={() => onPick(c.id)}
              className={`w-full rounded px-2 py-1.5 text-left transition ${
                activeId === c.id ? "bg-paper-300" : "hover:bg-paper-200"
              }`}
            >
              <div className="truncate text-[12px] text-ink-900">{c.title}</div>
              <div className="flex items-center gap-1.5 text-[10px] text-ink-400">
                <span>{c.n_turns} turns</span>
                {c.conversation_only_share != null && c.conversation_only_share > 0 && (
                  <span title="share of turns answered from the conversation rather than the documents">
                    {Math.round(c.conversation_only_share * 100)}% self
                  </span>
                )}
                {c.drifting && (
                  <span className="rounded bg-chart-600/10 px-1 text-chart-600">
                    drifting
                  </span>
                )}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ---------------------------------------------------------------------- app */

export function App() {
  const [screen, setScreen] = useState<Screen>("answers");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [activeConv, setActiveConv] = useState<string | null>(null);
  const [turns, setTurns] = useState<AskResponse[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"dense" | "sparse" | "rrf">("dense");
  const [budget, setBudget] = useState(1500);
  const [toast, setToast] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.status().then(setStatus).catch((e) => setErr(e.message));
    api.conversations().then((r) => setConvs(r.conversations)).catch(() => {});
  }, []);

  useEffect(refresh, [refresh]);

  const ask = async () => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setErr(null);
    try {
      let cid = activeConv;
      if (!cid) {
        cid = (await api.createConversation()).id;
        setActiveConv(cid);
      }
      const res = await api.askIn(cid, { question, budget, sources: 6, mode });
      setTurns((t) => [...t, res]);
      setQ("");
      api.conversations().then((r) => setConvs(r.conversations));
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  };

  const openConv = async (id: string) => {
    setActiveConv(id);
    setTurns([]);
    setScreen("answers");
    // The stored turns are the record of what was actually shown, including
    // citations into documents that may since have been removed.
    const c = await api.conversation(id);
    setTurns(
      (c.turns ?? []).map((t: any) => ({
        question: t.question,
        understanding: {
          route: t.route, header: t.header, read_as: t.read_as,
          search_query: t.search_query, was_rewritten: t.was_rewritten,
          needs_retrieval: t.retrieval_ran, original: t.question,
          topic_shift: false, suggested_fresh_question: "", fallback: false,
        },
        answer_markdown: t.answer_markdown,
        abstained: t.abstained, abstain_reason: "", grounded: t.grounded, reasons: [],
        claims: t.claims ?? [], evidence_mix: t.evidence_mix ?? {},
        reconciliation: t.reconciliation ?? {}, citation_integrity: {} as any,
        conversation_attribution: "per_claim", sources: t.sources ?? [],
        retrieval: {
          mode: "-", budget: 0, ran: t.retrieval_ran, skipped_because: null,
          children_considered: 0, parents_delivered: (t.sources ?? []).length,
          child_tokens: 0, parent_tokens: 0, leg_stats: null, fusion_explain: null,
        },
        usage: { prompt_tokens: 0, output_tokens: 0, cached_tokens: 0 },
        timings_ms: t.timings_ms ?? {},
      })) as AskResponse[],
    );
  };

  const removeDoc = async (id: string) => {
    const r = await api.removeDocument(id);
    setToast(
      `Removing ${id}: ${r.will_remove.chunks} passages, ${r.will_remove.cache_entries} cache entries. ${r.note}`,
    );
    const poll = async () => {
      const j = await api.job(r.job.id);
      if (j.state === "queued" || j.state === "running") setTimeout(poll, 1500);
      else refresh();
    };
    poll();
  };

  const last = turns[turns.length - 1];

  return (
    <div className="grid h-full grid-cols-[17rem_1fr] overflow-hidden">
      {/* sidebar */}
      <aside className="flex h-full flex-col overflow-y-auto border-r border-paper-400 bg-paper-100 px-4 py-4">
        <div>
          <h1 className="font-serif text-[19px] leading-none text-ink-900">RAGkit</h1>
          <p className="mt-1 text-[11px] text-ink-500">
            answers from your documents, with receipts
          </p>
        </div>

        <nav className="mt-4 flex gap-1">
          {(["answers", "inspector"] as Screen[]).map((s) => (
            <button
              key={s}
              onClick={() => setScreen(s)}
              className={`rounded px-2 py-1 text-[11px] uppercase tracking-wide transition ${
                screen === s
                  ? "bg-ink-900 text-paper-50"
                  : "text-ink-500 hover:bg-paper-200"
              }`}
            >
              {s}
            </button>
          ))}
        </nav>

        <div className="mt-5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-500">
            Your documents
          </div>
          <div className="mt-2">
            <DocumentList
              status={status}
              onRemove={removeDoc}
              readOnly={!!status?.demo?.read_only}
            />
          </div>
          <div className="mt-3">
            <Uploader
              onDone={refresh}
              /* uploads_enabled, NOT read_only. The demo refuses deletion and
                 accepts uploads; driving both controls off one flag is what
                 made "turn on uploads" look like a one-line change. */
              readOnly={status?.demo?.uploads_enabled === false}
              readOnlyWhy={status?.demo?.why ?? ""}
            />
          </div>
        </div>

        <div className="mt-5">
          <ConversationList
            convs={convs}
            activeId={activeConv}
            onPick={openConv}
            onNew={() => {
              setActiveConv(null);
              setTurns([]);
            }}
          />
        </div>

        <div className="mt-auto pt-5">
          {status?.degradations?.length ? (
            <DegradationBanner items={status.degradations} />
          ) : null}
          <p className="mt-3 text-[10px] leading-snug text-ink-400">{status?.scope_note}</p>
          {status?.index?.pipeline_fingerprint && (
            <p className="mt-1.5 font-mono text-[10px] text-ink-400">
              {status.index.n_children ?? status.index.children} passages ·{" "}
              {status.index.pipeline_fingerprint}
            </p>
          )}
        </div>
      </aside>

      {/* main */}
      <main className="flex h-full flex-col overflow-hidden">
        {screen === "inspector" ? (
          <Inspector />
        ) : (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto">
              {turns.length === 0 ? (
                <div className="mx-auto max-w-2xl px-6 py-12">
                  <h2 className="font-serif text-[22px] text-ink-900">
                    Answers from your documents, with receipts.
                  </h2>
                  <p className="mt-2 text-[13px] leading-relaxed text-ink-500">
                    There is no general knowledge here. If it isn't in what you add,
                    the answer is “not in your documents.”
                  </p>
                  <p className="mt-4 text-[12px] leading-relaxed text-ink-500">
                    Every claim links to where it came from —{" "}
                    <span className="text-quote-600">blue</span> is quoted text,{" "}
                    <span className="text-chart-600">amber</span> is a chart the
                    assistant read for you, and grey means it came from this
                    conversation rather than a document.
                  </p>
                </div>
              ) : (
                <div className="h-full">
                  <AnswerView
                    turns={turns}
                    conversationId={activeConv ?? undefined}
                    toast={toast}
                    onToast={setToast}
                  />
                </div>
              )}
            </div>

            {/* composer */}
            <div className="border-t border-paper-400 bg-paper-100 px-6 py-3">
              {err && <div className="mb-2 text-[12px] text-red-800">{err}</div>}
              <div className="flex items-end gap-2">
                <textarea
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      ask();
                    }
                  }}
                  rows={2}
                  placeholder={
                    status?.documents.length
                      ? "Ask a question about your documents…"
                      : "Add a document first"
                  }
                  className="min-h-[2.75rem] flex-1 resize-none rounded-md border border-paper-400 bg-paper-50 px-3 py-2 font-serif text-[14px] text-ink-900 outline-none placeholder:text-ink-400 focus:border-ink-400"
                />
                <button
                  onClick={ask}
                  disabled={busy || !q.trim()}
                  className="rounded-md bg-ink-900 px-3 py-2 text-[12px] text-paper-50 disabled:opacity-40"
                >
                  {busy ? "…" : "Ask"}
                </button>
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-ink-400">
                <label className="flex items-center gap-1">
                  retrieval
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value as any)}
                    className="rounded border border-paper-400 bg-paper-50 px-1 py-0.5"
                  >
                    <option value="dense">dense</option>
                    <option value="sparse">BM25</option>
                    <option value="rrf">RRF</option>
                  </select>
                </label>
                <label className="flex items-center gap-1">
                  budget
                  <input
                    type="number"
                    value={budget}
                    step={500}
                    min={250}
                    max={12000}
                    onChange={(e) => setBudget(Number(e.target.value))}
                    className="w-20 rounded border border-paper-400 bg-paper-50 px-1 py-0.5 tabular-nums"
                  />
                  tokens
                </label>
                {turns.length > 0 && last?.conversation?.drift && (
                  <span
                    title={last.conversation.drift.note}
                    className={
                      last.conversation.drift.drifting ? "text-chart-600" : undefined
                    }
                  >
                    {last.conversation.drift.conversation_only} of{" "}
                    {last.conversation.drift.turns} turns answered from the
                    conversation
                    {last.conversation.drift.drifting && " · drifting"}
                  </span>
                )}
                <span className="ml-auto">
                  query condensation is on — pronoun follow-ups are rewritten
                </span>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
