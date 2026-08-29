/**
 * The Answers screen. Every visual distinction here is a stored field, not a
 * judgement made in the browser.
 *
 * WHAT THE COLOURS MEAN, and why they are trustworthy:
 *   blue   quoted             verified verbatim against the chunk
 *   stone  structure_inferred text_source = page_text_clip (repaired table)
 *   amber  assistant_reading  text_provenance = MODEL_GENERATED (a caption)
 *   grey   conversation       the model declared the claim conversation-derived
 *
 * The backend derives each from provenance and sends the label with it. If this
 * component inspected the text and guessed, amber would mean "looks like a
 * chart" instead of "a model read a chart", and the whole point would be lost.
 *
 * THE WHOLE CONVERSATION IS RENDERED, not just the newest turn. The first
 * version showed `turns[turns.length - 1]` only, which loses the thing the
 * design is actually about: each turn carries its own route header, so a reader
 * can see grounding CHANGE across a conversation. A single turn cannot show
 * drift, and drift is the degradation this project exists to make visible.
 */
import { useState } from "react";
import {
  api,
  EVIDENCE,
  type AskResponse,
  type Citation,
  type Claim,
  type SourceDetail,
} from "../api";
import { Panel, Spinner } from "./primitives";
import Markdown, { Inline } from "./Markdown";
import { useEffect, useRef } from "react";

/* -------------------------------------------------------------- citation chip */

function Chip({ cit, onOpen }: { cit: Citation; onOpen: (c: Citation) => void }) {
  if (cit.fabricated) {
    return (
      <span
        className="ml-1 inline-flex items-baseline rounded border border-red-700/40 bg-red-700/5 px-1 align-baseline text-[10px] font-medium text-red-800"
        title={cit.detail}
      >
        [{cit.label}] fabricated
      </span>
    );
  }
  const ev = EVIDENCE[cit.evidence_kind ?? "found_not_quoted"];
  return (
    <button
      onClick={() => onOpen(cit)}
      title={`${ev.long}${cit.detail ? `\n${cit.detail}` : ""}`}
      className={`ml-1 inline-flex items-baseline gap-1 rounded border px-1 align-baseline text-[10px] font-medium transition hover:brightness-95 ${ev.chip}`}
    >
      <span className={`size-1.5 shrink-0 rounded-full ${ev.dot}`} />[{cit.label}]
    </button>
  );
}

/* ------------------------------------------------------------- flag dialogue */

function FlagBox({
  target,
  conversationId,
  turnIndex,
  onClose,
  onSent,
}: {
  target: { claim: Claim; index: number; cit?: Citation };
  conversationId?: string;
  turnIndex: number;
  onClose: () => void;
  onSent: (label: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  // The two options are the two DIFFERENT citation failures, which have
  // different fixes: a fabricated reference versus a real reference attached to
  // an unsupported claim. `source_is_fine` is the third, and it is the only way
  // the table detector's false-positive rate can ever be observed -- that number
  // is not derivable from the detector.
  const opts: { v: any; label: string; hint: string }[] = [
    { v: "not_in_source", label: "This isn't in the source", hint: "the cited passage does not contain it" },
    { v: "source_doesnt_say", label: "The source doesn't say this", hint: "the passage exists but does not support the claim" },
    { v: "source_is_fine", label: "This source reads fine to me", hint: "we flagged or declined it and we were wrong" },
  ];
  return (
    <div className="mt-2 rounded-md border border-paper-400 bg-paper-100 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
        What's wrong?
      </div>
      <div className="mt-2 flex flex-col gap-1.5">
        {opts.map((o) => (
          <button
            key={o.v}
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try {
                const r = await api.flag({
                  verdict: o.v,
                  conversation_id: conversationId,
                  turn_index: turnIndex,
                  claim_index: target.index,
                  claim_text: target.claim.text,
                  chunk_id: target.cit?.chunk_id ?? "",
                });
                onSent(r.label);
              } catch (e: any) {
                onSent(`could not record: ${e.message}`);
              } finally {
                setBusy(false);
                onClose();
              }
            }}
            className="rounded border border-paper-400 bg-paper-50 px-2 py-1.5 text-left text-[12px] text-ink-900 hover:border-ink-400"
          >
            {o.label}
            <span className="block text-[11px] text-ink-400">{o.hint}</span>
          </button>
        ))}
      </div>
      <button onClick={onClose} className="mt-2 text-[11px] text-ink-400 hover:text-ink-600">
        cancel
      </button>
    </div>
  );
}

/* --------------------------------------------------------------------- a turn */

function TurnBlock({
  res,
  turnIndex,
  conversationId,
  onOpen,
  onToast,
}: {
  res: AskResponse;
  turnIndex: number;
  conversationId?: string;
  onOpen: (chunkId: string, quote: string | null) => void;
  onToast: (s: string) => void;
}) {
  const [flag, setFlag] = useState<{ claim: Claim; index: number; cit?: Citation } | null>(
    null,
  );
  const u = res.understanding;
  const r = res.retrieval;
  const recon = res.reconciliation ?? {};

  return (
    <article className="border-b border-paper-300 px-6 py-5 last:border-b-0">
      <p className="font-sans text-[13px] font-medium text-ink-500">{res.question}</p>

      {/* The route header, before a word of the answer: where it came from is
          declared rather than left for the reader to infer. */}
      <div className="mt-2 flex flex-wrap items-baseline gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-500">
          {u.header}
        </span>
        <span className="text-[11px] text-ink-400">
          {r.ran
            ? `${r.parents_delivered} passage${r.parents_delivered === 1 ? "" : "s"} read`
            : `nothing new was read${r.skipped_because ? ` (${r.skipped_because})` : ""}`}
        </span>
      </div>
      {u.read_as && (
        <div className="mt-0.5 text-[12px] italic text-ink-500">read as: {u.read_as}</div>
      )}
      {u.was_rewritten && (
        <div className="text-[11px] text-ink-400">searched as “{u.search_query}”</div>
      )}
      {u.fallback && (
        <div className="text-[11px] text-chart-600">
          rewriting was unavailable — searched as typed
        </div>
      )}

      {!res.grounded && (
        <div className="mt-3 rounded-md border border-red-700/40 bg-red-700/5 px-3 py-2">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-red-800">
            Ungrounded — answer withheld
          </div>
          <ul className="mt-1 list-inside list-disc text-[12px] text-ink-600">
            {res.reasons.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        </div>
      )}

      {res.abstained ? (
        <div className="mt-3 rounded-md border border-paper-400 bg-paper-100 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Not answered
          </div>
          <p className="mt-1 font-serif text-[14px] leading-relaxed text-ink-900">
            {res.abstain_reason || "the sources do not contain this"}
          </p>
          {res.suggested_question && (
            // Comes from the ABSTENTION now, not the condenser's ask_fresh
            // route -- that route never fires, so the affordance was attached to
            // a branch that does not execute.
            <p className="mt-2 text-[12px] text-quote-600">
              Your documents may answer this if asked directly — try “
              {res.suggested_question}”
            </p>
          )}
        </div>
      ) : res.claims.length > 0 ? (
        <ul className="mt-3 space-y-0.5">
          {res.claims.map((claim, i) => {
            const conv = claim.source === "conversation";
            return (
              <li
                key={i}
                className={`group relative py-1.5 pl-3 ${
                  conv
                    ? "border-l-2 border-dashed border-ink-400/50"
                    : "border-l-2 border-paper-400"
                }`}
              >
                <p className="font-serif text-[15px] leading-relaxed text-ink-900">
                  <Inline text={claim.text} />
                  {claim.citations.map((c, j) => (
                    <Chip
                      key={j}
                      cit={c}
                      onOpen={(cit) => cit.chunk_id && onOpen(cit.chunk_id, cit.quote)}
                    />
                  ))}
                  {conv && (
                    <span className="ml-1.5 rounded bg-paper-200 px-1 text-[10px] uppercase tracking-wide text-ink-400">
                      this conversation
                    </span>
                  )}
                </p>
                <button
                  onClick={() => setFlag({ claim, index: i, cit: claim.citations[0] })}
                  className="absolute right-0 top-1.5 rounded px-1.5 py-0.5 text-[10px] text-ink-400 opacity-0 transition group-hover:opacity-100 hover:bg-paper-200 hover:text-ink-600"
                >
                  flag
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        // NO CLAIMS TO SHOW. Either the model abstained, or its structured
        // output was cut off at the output limit and only the prose survived.
        // This is the path where the field's name finally means something: it
        // was always called answer_markdown and was always rendered as plain
        // text, so `### **Profile & Menu**` reached the user verbatim.
        <Markdown text={res.answer_markdown} />
      )}

      {flag && (
        <FlagBox
          target={flag}
          conversationId={conversationId}
          turnIndex={turnIndex}
          onClose={() => setFlag(null)}
          onSent={(l) => onToast(`Noted — thanks. Recorded as “${l}”.`)}
        />
      )}

      {/* Reconciliation, per turn. Three counts that must agree, and
          `verification` when the checks did not apply at all. */}
      <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-ink-400 tabular-nums">
        <span>sent {recon.chunks_sent ?? 0}</span>
        <span>cited {recon.chunks_cited ?? 0}</span>
        <span>verified {recon.citations_verified ?? 0}</span>
        {(recon.citations_unquotable as number) > 0 && (
          <span>unquotable {recon.citations_unquotable as number}</span>
        )}
        {(recon.citations_fabricated as number) > 0 && (
          <span className="text-red-800">
            fabricated {recon.citations_fabricated as number}
          </span>
        )}
        {recon.verification === "not_applicable_no_sources" && (
          <span title="no documents were read, so the citation checks could not run — this is not a passing check">
            verification n/a
          </span>
        )}
        {Object.entries(res.timings_ms ?? {}).map(([k, v]) => (
          <span key={k}>
            {k} {Math.round(v)}ms
          </span>
        ))}
      </div>
    </article>
  );
}

/* -------------------------------------------------------------- source panel */

function SourcePanel({
  chunkId,
  quote,
  onClose,
}: {
  chunkId: string;
  quote: string | null;
  onClose: () => void;
}) {
  const [d, setD] = useState<SourceDetail | null>(null);
  const [err, setErr] = useState<{ msg: string; detail?: any } | null>(null);

  useEffect(() => {
    setD(null);
    setErr(null);
    api
      .source(chunkId, quote)
      .then(setD)
      .catch((e) => setErr({ msg: e.message, detail: e.detail }));
  }, [chunkId, quote]);

  const removed = err?.detail?.reason === "source_removed";

  return (
    <aside className="flex h-full flex-col overflow-hidden border-l border-paper-400 bg-paper-50">
      <header className="flex items-center justify-between border-b border-paper-300 px-4 py-3">
        <button onClick={onClose} className="text-[12px] text-ink-500 hover:text-ink-900">
          ← Back to the answer
        </button>
      </header>
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {!d && !err && <Spinner label="loading source" />}

        {removed && (
          // 410, not 404. "This was removed" and "this never existed" are
          // different answers, and collapsing them would make a correct old
          // answer look like a fabricated citation.
          <div className="rounded-md border border-dashed border-ink-400/50 bg-paper-200 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
              Source removed
            </div>
            <p className="mt-1.5 text-[13px] leading-snug text-ink-600">
              {err?.detail?.message}
            </p>
            <p className="mt-1.5 text-[11px] text-ink-400">
              {err?.detail?.source_id} · deleted {err?.detail?.deleted_at}
            </p>
          </div>
        )}
        {err && !removed && <div className="text-[13px] text-red-800">{err.msg}</div>}

        {d && (
          <>
            <div className="text-[11px] uppercase tracking-wide text-ink-400">{d.locator}</div>
            {d.heading_path.length > 0 && (
              <div className="mt-1 text-[12px] text-ink-500">{d.heading_path.join(" › ")}</div>
            )}
            <div
              className={`mt-3 inline-flex items-center gap-1.5 rounded border px-2 py-1 text-[11px] font-medium ${
                EVIDENCE[d.evidence_kind].chip
              }`}
            >
              <span className={`size-1.5 rounded-full ${EVIDENCE[d.evidence_kind].dot}`} />
              {d.evidence_label}
            </div>
            {d.note && <p className="mt-2 text-[12px] leading-snug text-ink-500">{d.note}</p>}
            {(d.table_header_missing || d.table_continuation_suspect) && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {d.table_header_missing && (
                  <span className="rounded bg-paper-300 px-1.5 py-0.5 text-[10px] text-ink-600">
                    header row not detected
                  </span>
                )}
                {d.table_continuation_suspect && (
                  <span className="rounded bg-paper-300 px-1.5 py-0.5 text-[10px] text-ink-600">
                    may continue from the previous page
                  </span>
                )}
              </div>
            )}
            {d.asset_url && (
              <img
                src={d.asset_url}
                alt={d.locator}
                className="mt-3 w-full rounded border border-paper-400"
              />
            )}
            <div className="mt-4">
              <div className="mb-1.5 text-[11px] uppercase tracking-wide text-ink-400">
                {d.highlight
                  ? "Highlighted — the exact span this answer cites"
                  : d.evidence_kind === "assistant_reading"
                    ? "The assistant's description — its words, not the document's"
                    : "The page's text, as extracted"}
              </div>
              <pre className="whitespace-pre-wrap break-words rounded border border-paper-300 bg-paper-100 p-3 font-serif text-[13px] leading-relaxed text-ink-900">
                {d.highlight ? (
                  <>
                    {d.text.slice(0, d.highlight.start)}
                    <mark className="rounded bg-quote-600/15 px-0.5 text-quote-600">
                      {d.text.slice(d.highlight.start, d.highlight.end)}
                    </mark>
                    {d.text.slice(d.highlight.end)}
                  </>
                ) : (
                  d.text
                )}
              </pre>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------ evidence summary */

function EvidenceSummary({ res }: { res: AskResponse }) {
  const groups = (Object.keys(EVIDENCE) as (keyof typeof EVIDENCE)[])
    .map((k) => ({
      k,
      n: res.evidence_mix?.[k] ?? 0,
      cits: res.claims.flatMap((c) => c.citations.filter((x) => x.evidence_kind === k)),
    }))
    .filter((g) => g.n > 0 || g.cits.length > 0);

  return (
    <Panel title="Evidence in the latest answer">
      {groups.length === 0 && (
        <p className="text-[12px] text-ink-400">
          nothing in this answer carries document evidence
        </p>
      )}
      <ul className="space-y-2.5">
        {groups.map((g) => (
          <li key={g.k}>
            <div className="flex items-start gap-1.5">
              <span className={`mt-1.5 size-1.5 shrink-0 rounded-full ${EVIDENCE[g.k].dot}`} />
              <div>
                <span className="text-[12px] font-medium text-ink-900">
                  {EVIDENCE[g.k].long}
                </span>
                <span className="ml-1.5 text-[11px] text-ink-400">
                  {g.n || g.cits.length}
                </span>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

/* ------------------------------------------------------------------- the view */

export function AnswerView({
  turns,
  conversationId,
  toast,
  onToast,
  pending,
}: {
  turns: AskResponse[];
  conversationId?: string;
  toast: string | null;
  onToast: (s: string | null) => void;
  // A question that has been sent and has no answer yet.
  pending?: string | null;
}) {
  const [open, setOpen] = useState<{ chunkId: string; quote: string | null } | null>(null);
  const last = turns[turns.length - 1];

  // THE SCROLLING COLUMN IS THIS ONE, and the first attempt put the listener on
  // the parent -- which never scrolls, because the transcript scrolls inside
  // here. A scroll handler attached to an element that cannot scroll is silent
  // rather than wrong, which is why it would have shipped.
  const col = useRef<HTMLDivElement>(null);
  const tail = useRef<HTMLDivElement>(null);
  const following = useRef(true);

  useEffect(() => {
    const el = col.current;
    if (!el) return;
    const onScroll = () => {
      following.current = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // ONLY IF THEY WERE FOLLOWING ALONG. This product invites you to scroll up
  // into a source and check a quotation; yanking the view away mid-read is
  // worse than not scrolling at all. And it scrolls to the START of the new
  // turn rather than the end of the page -- an answer is read from its
  // beginning, and landing at its end means scrolling back up to read it.
  useEffect(() => {
    if (!following.current) return;
    tail.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [turns.length, pending]);

  return (
    <div className="grid h-full grid-cols-1 overflow-hidden lg:grid-cols-[1fr_26rem]">
      <div ref={col} className="overflow-y-auto">
        {turns.map((t, i) => (
          <TurnBlock
            key={i}
            res={t}
            turnIndex={i}
            conversationId={conversationId}
            onOpen={(chunkId, quote) => setOpen({ chunkId, quote })}
            onToast={onToast}
          />
        ))}
        {/* The sent question, before its answer exists. Deliberately NOT
            shaped like a turn: it has no citations, no evidence and no
            reconciliation, and giving it that shape would imply those are
            pending rather than absent. */}
        {pending && (
          <div className="px-6 pb-8">
            <div className="border-t border-paper-300 pt-5">
              <p className="font-serif text-[16px] leading-snug text-ink-900">
                {pending}
              </p>
              <p className="mt-2 flex items-center gap-2 text-[12px] text-ink-400">
                <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-quote-600" />
                searching your documents…
              </p>
            </div>
          </div>
        )}
        {toast && (
          <div className="mx-6 mb-4 rounded border border-paper-400 bg-paper-100 px-3 py-2 text-[12px] text-ink-600">
            {toast}
          </div>
        )}
        {/* The scroll target, at the very end of the column. */}
        <div ref={tail} />
      </div>

      {open ? (
        <SourcePanel chunkId={open.chunkId} quote={open.quote} onClose={() => setOpen(null)} />
      ) : (
        <div className="overflow-y-auto border-l border-paper-400 bg-paper-100/60 px-4 py-5">
          {last && <EvidenceSummary res={last} />}
          {last && last.sources.length > 0 && (
            <Panel title="Sources read" className="mt-4">
              <ul className="space-y-2">
                {last.sources.map((s) => (
                  <li key={s.chunk_id}>
                    <button
                      onClick={() => setOpen({ chunkId: s.chunk_id, quote: null })}
                      className="w-full text-left"
                    >
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-[11px] text-ink-400">[{s.label}]</span>
                        <span className="text-[12px] text-ink-900 hover:underline">
                          {s.source_id}
                        </span>
                        <span className="text-[11px] text-ink-400">
                          {s.page_label ?? (s.page ? `p${s.page}` : "")}
                        </span>
                      </div>
                      {s.heading_path.length > 0 && (
                        <div className="truncate text-[11px] text-ink-400">
                          {s.heading_path.join(" › ")}
                        </div>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
          {last?.retrieval?.fusion_explain && (
            <Panel
              title="Fusion"
              subtitle="per-leg ranks — a fusion you cannot explain is one you cannot tune"
              className="mt-4"
            >
              <pre className="overflow-x-auto text-[11px] leading-relaxed text-ink-600">
                {last.retrieval.fusion_explain}
              </pre>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}
