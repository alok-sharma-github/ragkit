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
  type Doc,
  type Job,
  type StatusResponse,
} from "./api";
import { AnswerView } from "./components/Answer";
import { Inspector } from "./components/Inspector";
import { DegradationBanner, Spinner } from "./components/primitives";

type Screen = "answers" | "inspector";

/* ------------------------------------------------------------------- sidebar */

/** One sentence naming what the preloaded corpus IS, derived not hardcoded.
 *
 * Derived, because a hardcoded description becomes a lie the moment somebody
 * changes the corpus -- and the description of a document set is exactly the
 * kind of claim this project refuses to state without a source. It counts what
 * is actually indexed and says so; only the SUBJECT is a fixed string, and that
 * is the one part a file listing genuinely cannot know.
 */
function corpusBlurb(status: StatusResponse | null): string {
  if (!status) return "";
  const n = status.documents.filter((d) => d.doc_type !== "image").length;
  if (!n) return "Nothing is loaded yet — add a document to begin.";
  return `${n} papers and manuals on retrieval systems, already indexed — ` +
    `ask a question without uploading anything.`;
}

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

  // PAGE RENDERS BELONG TO THEIR PDF, not beside it.
  //
  // `assets/lost-in-the-middle_p1.png` is a page image extracted FROM
  // lost-in-the-middle.pdf. Listed as a peer it reads as a separate document, so
  // the corpus opened with four PNGs -- alphabetically first, least meaningful,
  // and occupying the top of the only column a visitor reads. Folding them under
  // their source is not cosmetic: a list that misrepresents what the corpus
  // CONTAINS is the same class of error as a metric that misrepresents what was
  // measured.
  const parentOf = (d: Doc): string | null => {
    if (d.doc_type !== "image") return null;
    const stem = d.source_id.split("/").pop()?.replace(/\.[^.]+$/, "") ?? "";
    const base = stem.replace(/_p\d+$/, "");
    const owner = status.documents.find(
      (o) => o.doc_type !== "image" && o.source_id.replace(/\.[^.]+$/, "") === base,
    );
    return owner ? owner.source_id : null;
  };
  const renders = new Map<string, Doc[]>();
  const top: Doc[] = [];
  for (const d of status.documents) {
    const owner = parentOf(d);
    if (owner) renders.set(owner, [...(renders.get(owner) ?? []), d]);
    else top.push(d);
  }
  const nRenders = status.documents.length - top.length;

  return (
    <div>
      <div className="text-[11px] text-ink-400">
        {top.length} documents
        {nRenders ? ` · ${nRenders} page images` : ""}
      </div>
      <ul className="mt-2 space-y-1.5">
        {top.map((d) => (
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
                  {renders.get(d.source_id)?.length
                    ? ` · ${renders.get(d.source_id)!.length} page images`
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
  limits,
}: {
  onDone: () => void;
  readOnly: boolean;
  readOnlyWhy: string;
  // The server's own sentence about what it accepts and how long it keeps it.
  // Rendered rather than restated, so the limit a visitor reads and the limit
  // that refuses them cannot drift apart.
  limits: string;
}) {
  // THE POLLER IS BACK, and the reason is measured rather than anticipated.
  //
  // It was removed when upload started indexing inline -- correct for a
  // two-page test file, wrong for a real one. A 14-page PDF takes minutes at
  // 0.25 vCPU and the proxy gives up at 60 seconds, so the visitor got a 504 on
  // an upload that was still working. The request now returns a job id and this
  // watches it.
  const [job, setJob] = useState<Job | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Polls until the job settles. The interval is deliberately unhurried: the
  // work takes minutes, and a tight poll would spend the visitor's rate-limit
  // allowance on status checks.
  const poll = useCallback((id: string) => {
    const tick = async () => {
      try {
        const j = await api.job(id);
        setJob(j);
        if (j.state === "queued" || j.state === "running") {
          setTimeout(tick, 2000);
          return;
        }
        setBusy(false);
        if (j.state === "failed") {
          setMsg(j.error?.split(String.fromCharCode(10))[0] ?? "could not read that document");
        } else {
          setNote("ready — ask a question about it");
          onDone();
        }
      } catch {
        setBusy(false);
        setMsg("lost contact with the server while reading your document");
      }
    };
    setTimeout(tick, 1500);
  }, [onDone]);

  const send = async (files: File[]) => {
    setMsg(null);
    setNote(null);
    setBusy(true);
    try {
      const up: any = await api.upload(files);
      if (up.rejected?.length) {
        setMsg(up.rejected.map((r: any) => `${r.name}: ${r.reason}`).join(" · "));
      }
      if (up.saved?.length && up.job) {
        setJob(up.job);
        poll(up.job.id);
      }
    } catch {
      setMsg("upload failed — check the file and try again");
      setBusy(false);
    }
    // `busy` is NOT cleared here: the upload request returning is the START of
    // the work, not the end of it. The poller clears it when the job settles.
  };

  // A DROP ZONE, not a button. Upload is the action that turns a demo corpus
  // into the visitor's own tool, so it gets the affordance that says "put your
  // file here" rather than one that says "browse".
  //
  // READ-ONLY DEPLOYMENT: replace the control, do not disable it. A dropzone
  // that accepts a file and then 403s is worse than no dropzone -- the user has
  // already committed before learning it was never allowed.
  if (readOnly) {
    return (
      <div className="rounded-md border border-dashed border-paper-400 bg-paper-50 px-3 py-3">
        <div className="text-[11px] font-semibold text-ink-500">
          Uploads are off on this deployment
        </div>
        <div className="mt-1 text-[11px] leading-relaxed text-ink-400">
          {readOnlyWhy || "this deployment is read-only"}
        </div>
      </div>
    );
  }

  const running = busy || (job !== null && (job.state === "queued" || job.state === "running"));
  // The stage the job reports, shown verbatim. "understanding the document" is
  // a truer thing to show for ninety seconds than a percentage would be -- cost
  // per page varies enough that a progress bar would be wrong most of the time.
  const stage = job?.progress?.stage ?? "";
  const detail = job?.progress?.detail ?? "";

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const files = Array.from(e.dataTransfer.files);
          if (files.length) send(files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-md border border-dashed px-3 py-4 text-center transition ${
          over
            ? "border-quote-600 bg-quote-600/5"
            : "border-paper-400 bg-paper-50 hover:border-ink-400"
        }`}
      >
        <div className="text-[12px] font-semibold text-quote-600">
          {running ? "Reading your document…" : "Add your documents"}
        </div>
        <div className="mt-1 text-[11px] leading-relaxed text-ink-400">
          {running
            ? (stage
                ? `${stage}${detail ? ` — ${detail}` : ""}`
                : "uploading…")
            : "Drop a PDF here, or click to browse"}
        </div>
        {running && (
          <div className="mt-2 rounded-sm bg-quote-600/[0.06] px-2 py-1.5 text-[10.5px] leading-relaxed text-quote-600">
            {/* THE DESIGNED ANSWER TO THE WAIT, and it was never built:
                "you can ask questions while they prepare". A two-minute wait
                with nothing to do reads as broken; the same wait with something
                worth doing reads as work happening. And it is true -- the
                preloaded corpus is searchable throughout, because the index is
                only locked for the final append. */}
            You can ask questions of the loaded papers while this finishes —
            about a minute for every five pages.
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) send(files);
          e.target.value = "";
        }}
      />
      {/* THE LIMITS, BEFORE THEY REFUSE ANYTHING. A cap a visitor discovers by
          hitting it is indistinguishable from a bug. */}
      {limits && (
        <p className="mt-1.5 text-[10.5px] leading-relaxed text-ink-400">{limits}</p>
      )}
      {note && <p className="mt-1.5 text-[11px] text-quote-600">{note}</p>}
      {msg && <p className="mt-1.5 text-[11px] leading-relaxed text-chart-600">{msg}</p>}
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
  // The corpus list is collapsed by default: it is reference material, not
  // an action, and open-by-default is what pushed upload off-screen.
  const [docsOpen, setDocsOpen] = useState(false);
  const [advanced, setAdvanced] = useState(false);
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

        {/* UPLOAD FIRST, and the ordering is the point.
            The list of documents a visitor did NOT add was on top, sorted
            alphabetically, opening with four page images and filling the column
            -- so the one control that makes this THEIR tool sat below the fold.
            The most valuable action was invisible and the least actionable
            content was dominant. */}
        <div className="mt-5">
          <Uploader
            onDone={refresh}
            /* uploads_enabled, NOT read_only. The demo refuses deletion and
               accepts uploads; driving both controls off one flag is what
               made "turn on uploads" look like a one-line change. */
            readOnly={status?.demo?.uploads_enabled === false}
            readOnlyWhy={status?.demo?.why ?? ""}
            limits={status?.demo?.upload_limits ?? ""}
          />
        </div>

        {/* WHAT THIS CORPUS IS ABOUT, in one line. A visitor saw hnsw.pdf,
            hyde.pdf, raptor.pdf and had no way to know these are retrieval
            papers, or that the demo is preloaded so they can try it WITHOUT
            uploading anything. "15 documents · 4 image, 10 pdf" is an inventory;
            it says nothing about what the inventory is about. */}
        <div className="mt-5">
          <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-500">
            Already loaded
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-ink-500">
            {corpusBlurb(status)}
          </p>
          <button
            onClick={() => setDocsOpen((v) => !v)}
            className="mt-1.5 text-[11px] text-ink-400 hover:text-ink-900"
          >
            {docsOpen ? "hide the list" : "see the list"}
          </button>
          {/* SCROLLS WITHIN ITSELF. Collapsed by default, and even open it
              cannot push the upload control off-screen again. */}
          {docsOpen && (
            <div className="mt-2 max-h-64 overflow-y-auto pr-1">
              <DocumentList
                status={status}
                onRemove={removeDoc}
                readOnly={!!status?.demo?.read_only}
              />
            </div>
          )}
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
                /* SHOW THE BEHAVIOUR, THEN NAME THE NOTATION.
                   This screen used to open with a colour legend -- "blue is
                   quoted text, amber is a chart the assistant read for you" --
                   which is the key to a map nobody has been shown. It answered a
                   question the visitor had not asked yet. The worked example
                   below comes first, and the sentence explaining the colours
                   comes after the colours have done something. */
                <div className="mx-auto max-w-2xl px-6 py-12">
                  <h2 className="font-serif text-[22px] leading-snug text-ink-900">
                    Answers from your documents, with receipts.
                  </h2>
                  <p className="mt-2 text-[13px] leading-relaxed text-ink-500">
                    There is no general knowledge here. If it isn't in what you
                    add, the answer is “not in your documents.”
                  </p>

                  <div className="mt-7 rounded-md border border-paper-400 bg-paper-50 p-5">
                    <span className="rounded-sm border border-dashed border-paper-400 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-ink-400">
                      Example — not your documents yet
                    </span>
                    <p className="mt-3 text-[12px] text-ink-400">
                      What notice period does the services contract require?
                    </p>
                    <p className="mt-2 font-serif text-[15.5px] leading-[1.8] text-ink-900">
                      Either party may terminate with 60 days' written notice,
                      <span className="ml-1 align-super rounded-sm border border-quote-600/40 px-1 text-[10px] font-sans text-quote-600">
                        1
                      </span>{" "}
                      and the fee schedule steps down after year two.
                      <span className="ml-1 align-super rounded-sm border border-chart-600/45 bg-chart-600/[0.07] px-1 text-[10px] font-sans text-chart-600">
                        Fig 1
                      </span>
                    </p>
                    <p className="mt-2.5 text-[11.5px] leading-relaxed text-ink-400">
                      Every claim links to the exact place it came from —{" "}
                      <span className="text-quote-600">blue</span> is quoted text,{" "}
                      <span className="text-chart-600">amber</span> is a chart or
                      table the assistant read for you.
                    </p>
                    <div className="mt-3.5 border-t border-paper-300 pt-3">
                      <span className="rounded-sm border border-paper-400 px-1.5 py-0.5 text-[10.5px] font-semibold uppercase tracking-[0.1em] text-ink-600">
                        Found — not quoted
                      </span>
                      <p className="mt-2 font-serif text-[14.5px] text-ink-900">
                        “It's in your documents. I won't quote it.”
                      </p>
                      <p className="mt-1.5 text-[11.5px] text-ink-400">
                        When it can't verify something, it hands you the source
                        instead of a guess.
                      </p>
                    </div>
                  </div>

                  <p className="mt-6 text-[13px] leading-relaxed text-ink-500">
                    {status?.documents.length
                      ? "Ask a question below — the corpus on the left is already indexed. Or add your own document to see it answer from that instead."
                      : "Add a document on the left to begin."}
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
                  /* Never disabled while a document is indexing. The corpus is
                     searchable throughout, and taking the composer away during
                     the one wait in the product would remove the only thing
                     there is to do. */
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
              {/* DEVELOPER CONTROLS, BEHIND A TOGGLE.
                  `retrieval: dense | BM25 | RRF` and `budget: 1500 tokens` sat
                  permanently under the composer. A first-time visitor has no
                  basis for choosing between those, and 1500 is a number they
                  cannot reason about -- so the two most prominent controls on
                  the screen were ones only their author could use. They belong
                  to the Inspector's audience, not this one, and they stay
                  reachable because the comparison they enable is real. */}
              <div className="mt-1.5 flex flex-wrap items-center gap-3 text-[11px] text-ink-400">
                <button
                  onClick={() => setAdvanced((v) => !v)}
                  className="rounded px-1 text-ink-400 hover:bg-paper-200 hover:text-ink-900"
                >
                  {advanced ? "hide retrieval settings" : "retrieval settings"}
                </button>
                {advanced && (
                  <>
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
                  </>
                )}
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
