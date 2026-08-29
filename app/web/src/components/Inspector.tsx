/**
 * The Inspector. Its thesis is the backend's:
 *
 *     counts must reconcile before any score on this run is trusted
 *
 * Every number on this page inherits the pipeline fingerprint shown at the top.
 * A figure measured under a different parser, chunker or embedding model
 * describes a different system, so the fingerprint is not decoration -- it is
 * the scope of every claim below it.
 */
import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type InspectorResponse, type Rate } from "../api";
import { FingerprintChip, Panel, RateView, Spinner, StateBadge } from "./primitives";

const asRate = (v: unknown): Rate | undefined =>
  v && typeof v === "object" && "label" in (v as any) ? (v as Rate) : undefined;

/* ------------------------------------------------------------ reconciliation */

/** The one number, alone, with everything that qualifies it.
 *
 * THE PAGE OPENED ON A DETAIL. The headline -- 82/92 at a 1500-token budget on
 * five of seven strata -- was buried inside the "by stratum" card, so a reader
 * scrolled into a filesystem invariant mid-sentence and had to hunt for the
 * summary. A page whose most important number is discoverable rather than
 * stated is asking the reader to already know the argument.
 *
 * The qualifiers are attached to the number rather than printed near it,
 * because this project's whole position is that the number without them is a
 * different claim.
 */
function Headline({ d }: { d: InspectorResponse }) {
  const ev = d.eval;
  const h = ev?.headline;
  // headline carries both Rate objects and bare numbers (mean token counts), so
  // the one being rendered is narrowed rather than assumed.
  const cs = h && typeof h.child_strict === "object" ? h.child_strict : null;
  if (!ev || !cs) return null;
  const trusted = d.reconciliation.summary.failing === 0;
  return (
    <section className="rounded-md border border-ink-900/15 bg-paper-50 px-6 py-5 shadow-[0_1px_0_rgba(35,40,46,.04)]">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-500">
          Retrieval recall
        </div>
        <div
          className={`text-[10px] font-semibold uppercase tracking-[0.12em] ${
            trusted ? "text-quote-600" : "text-red-800"
          }`}
        >
          {trusted ? "counts reconcile — scores are trusted" : "counts do not reconcile"}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-5 gap-y-2">
        <div className="font-serif text-[40px] leading-none tabular-nums text-ink-900">
          {cs.label}
        </div>
        <div className="text-[12px] leading-relaxed text-ink-500">
          every fact needed to answer, inside the retrieved passages
        </div>
      </div>
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[11.5px] text-ink-500">
        <span>
          at a <strong className="text-ink-900">{ev.token_budget}-token</strong> budget
        </span>
        {/* scope_label starts by repeating the headline number, which is
            already the biggest thing on the card. Keep the qualifier, drop the
            restatement -- a figure printed twice reads as two figures. */}
        <span>{ev.scope_label.replace(/^[^ ]+ = [^ ]+ /, "")}</span>
        <span className="font-mono text-[10.5px] text-ink-400">
          {d.reconciliation.pipeline_fingerprint}
        </span>
      </div>
      <p className="mt-3 border-t border-paper-300 pt-2.5 text-[11.5px] leading-relaxed text-ink-500">
        Change the budget, the pipeline, or which text a passage is charged for,
        and this is a different number. That is why all three are printed beside
        it rather than in a footnote.
      </p>
    </section>
  );
}

/** All measured failures, by Barnett failure point.
 *
 * The strongest single result here, and it lived in a markdown file: every
 * failure is FP2 or FP3, and NONE are FP4. A zero in the right place is a
 * finding -- it rules out prompt engineering, reordering and reader
 * fine-tuning on evidence, because each improves a step that is not failing.
 *
 * The zeros are NOT all the same, and the card says which is which. FP1 cannot
 * be observed at all (the golden set is generated from the corpus). FP6 and FP7
 * need a human judgement nothing here performs. Only FP4's zero means "measured,
 * did not happen" -- and that is the one carrying the argument.
 */
function Failures({ f }: { f: any }) {
  if (!f?.points) return null;
  const max = Math.max(1, ...f.points.map((p: any) => p.n));
  return (
    <Panel
      title="Where it fails"
      subtitle="all measured failures, classified against Barnett's seven failure points"
    >
      <div className="space-y-1.5">
        {f.points.map((p: any) => {
          const measured = p.observable === "measured";
          return (
            <div key={p.id} className="flex items-center gap-3">
              <div className="w-[9.5rem] shrink-0 text-[11.5px] text-ink-900">
                <span className="font-mono text-[10.5px] text-ink-400">{p.id}</span>{" "}
                {p.name}
              </div>
              <div className="h-3 flex-1 rounded-sm bg-paper-200">
                <div
                  className="h-3 rounded-sm bg-ink-900/70"
                  style={{ width: `${(p.n / max) * 100}%` }}
                />
              </div>
              <div className="w-8 shrink-0 text-right text-[11.5px] tabular-nums text-ink-900">
                {p.n}
              </div>
              <div className="w-[13rem] shrink-0 text-[10.5px] leading-tight text-ink-400">
                {measured ? (p.n ? "" : "measured — did not happen") : p.observable}
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-3 border-t border-paper-300 pt-2.5 text-[11.5px] leading-relaxed text-ink-500">
        {f.thesis ??
          "every measured failure is FP2 or FP3 — the evidence exists and was not delivered. Zero FP4 is a negative result, not missing data: it rules out prompt engineering, context reordering and reader fine-tuning, because each improves a step that is not failing."}
      </p>
      <p className="mt-2 text-[11px] leading-relaxed text-ink-400">
        {f.n} failures at a {f.budget}-token budget. A zero is not one thing: FP1
        is structurally unobservable here, FP6 and FP7 need a human judgement
        nothing in this project performs, and only FP4's zero means it was
        measured and did not occur.
      </p>
    </Panel>
  );
}

function Reconciliation({ data }: { data: InspectorResponse["reconciliation"] }) {
  return (
    <Panel
      title="Reconciliation"
      subtitle={data.thesis}
      right={
        <div className="text-right">
          <div className="text-[12px] tabular-nums text-ink-600">
            <span className={data.summary.failing ? "text-red-800" : ""}>
              {data.summary.failing} failing
            </span>
            {" · "}
            {data.summary.passing} passing {" · "}
            {data.summary.not_measured} not measured
          </div>
          <div
            className={`mt-1 text-[11px] font-medium uppercase tracking-wide ${
              data.trusted ? "text-quote-600" : "text-red-800"
            }`}
          >
            {data.trusted ? "scores are trusted" : "scores are NOT trusted"}
          </div>
        </div>
      }
    >
      <FingerprintChip parts={data.fingerprint} hash={data.pipeline_fingerprint} />
      <p className="mt-1 text-[11px] text-ink-400">
        every figure on this page inherits this fingerprint
      </p>
      <ul className="mt-3 divide-y divide-paper-300">
        {data.checks.map((c) => (
          <li key={c.name} className="py-2.5">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium text-ink-900">{c.name}</span>
                  <StateBadge state={c.state} />
                  {c.n != null && (
                    <span className="text-[11px] text-ink-400 tabular-nums">n={c.n}</span>
                  )}
                </div>
                <div className="mt-0.5 text-[12px] text-ink-600">{c.observed}</div>
                <div className="mt-0.5 font-mono text-[11px] text-ink-400">
                  rule: {c.rule}
                </div>
                {c.detail && (
                  <div className="mt-1 text-[11px] leading-snug text-ink-500">{c.detail}</div>
                )}
                {c.why && (
                  <div className="mt-1 text-[11px] leading-snug text-ink-400">
                    exists because: {c.why}
                  </div>
                )}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

/* --------------------------------------------------------- recall vs budget */

function BudgetSweep({
  sweep,
}: {
  sweep: NonNullable<InspectorResponse["eval"]>["budget_sweep"];
}) {
  const rows = Object.entries(sweep ?? {}).map(([budget, h]) => ({
    budget: Number(budget),
    child: (asRate((h as any).child_strict)?.rate ?? null) as number | null,
    parent: (asRate((h as any).parent_strict)?.rate ?? null) as number | null,
    childTok: (h as any).mean_child_tokens as number,
    parentTok: (h as any).mean_parent_tokens as number,
    noneChild: (h as any).child_no_delivery as number,
    noneParent: (h as any).parent_no_delivery as number,
  }));

  return (
    <Panel
      title="Recall vs budget"
      subtitle="a single point hides where the curve bends — and where it bends is the only thing that says whether more retrieval would help"
    >
      <div className="h-56">
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 4, left: -18 }}>
            <CartesianGrid stroke="#e3dfd6" vertical={false} />
            <XAxis
              dataKey="budget"
              scale="log"
              domain={["dataMin", "dataMax"]}
              type="number"
              tick={{ fontSize: 11, fill: "#8a857a" }}
              tickFormatter={(v) => `${v}`}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 11, fill: "#8a857a" }}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <Tooltip
              contentStyle={{
                fontSize: 12,
                borderRadius: 6,
                border: "1px solid #d5d0c5",
                background: "#fbfaf7",
              }}
              formatter={(v: any, n) => [
                v == null ? "—" : `${Math.round(v * 100)}%`,
                n === "child" ? "child strict" : "parent strict",
              ]}
              labelFormatter={(l) => `${l} tokens`}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line
              type="monotone"
              dataKey="child"
              name="child strict"
              stroke="#2c4a75"
              strokeWidth={2}
              dot={{ r: 2.5 }}
            />
            <Line
              type="monotone"
              dataKey="parent"
              name="parent strict"
              stroke="#a8792f"
              strokeWidth={2}
              dot={{ r: 2.5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <table className="mt-2 w-full text-[11px] tabular-nums">
        <thead className="text-ink-400">
          <tr className="text-left">
            <th className="py-1 font-medium">budget</th>
            <th className="font-medium">child tok</th>
            <th className="font-medium">parent tok</th>
            <th className="font-medium" title="items where the unit could deliver nothing within the budget">
              nothing fit (c/p)
            </th>
          </tr>
        </thead>
        <tbody className="text-ink-600">
          {rows.map((r) => (
            <tr key={r.budget} className="border-t border-paper-300">
              <td className="py-1">{r.budget}</td>
              <td>{r.childTok}</td>
              <td>{r.parentTok}</td>
              <td>
                {r.noneChild}/{r.noneParent}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1.5 text-[11px] leading-snug text-ink-400">
        “nothing fit” is not a ranking failure — a 1200-token parent is
        undeliverable inside 250 tokens. Counted apart so the two causes stay legible.
      </p>
    </Panel>
  );
}

/* --------------------------------------------------------------- strata table */

function Strata({ ev }: { ev: NonNullable<InspectorResponse["eval"]> }) {
  const cov = ev.stratum_coverage;
  return (
    <Panel
      title="By stratum"
      subtitle={ev.scope_label}
      right={
        <div className="text-right text-[11px] text-ink-400">
          budget {ev.token_budget} · {ev.golden_set.evaluable} items
        </div>
      }
    >
      <table className="w-full text-[12px]">
        <thead className="text-[11px] text-ink-400">
          <tr className="text-left">
            <th className="py-1 font-medium">stratum</th>
            <th className="font-medium">child strict</th>
            <th className="font-medium">parent strict</th>
          </tr>
        </thead>
        <tbody>
          {cov.declared.map((st) => {
            const row = ev.by_stratum[st];
            const missing = cov.missing.includes(st);
            // THREE STATES, not two. `out_of_scope` is declared, present in the
            // golden set, and deliberately absent from `by_stratum`: its correct
            // behaviour is ABSTENTION, which is a generation-tier property, so
            // scoring it as retrieval recall would penalise the retriever for
            // the eval's own design. Rendering it as an em-dash made it look
            // like missing data, which is the same conflation the whole
            // NOT_MEASURED badge exists to prevent.
            const otherTier = !missing && !row;
            return (
              <tr key={st} className="border-t border-paper-300">
                <td className="py-1.5">
                  <span className={missing ? "text-ink-400" : "text-ink-900"}>{st}</span>
                  {missing && (
                    <span className="ml-2 rounded border border-dashed border-ink-400/50 px-1 text-[10px] uppercase tracking-wide text-ink-400">
                      not measured
                    </span>
                  )}
                  {cov.thin.includes(st) && (
                    <span className="ml-2 rounded border border-ink-400/30 px-1 text-[10px] uppercase tracking-wide text-ink-400">
                      thin
                    </span>
                  )}
                </td>
                {otherTier ? (
                  <td
                    colSpan={2}
                    className="text-[11px] text-ink-400"
                    title="correct behaviour here is abstention, which the retrieval tier cannot score"
                  >
                    scored in the generation tier — abstention, not recall
                  </td>
                ) : (
                  <>
                    <td className="text-ink-600">
                      {row ? <RateView rate={asRate(row.child_strict)} /> : "—"}
                    </td>
                    <td className="text-ink-600">
                      {row ? <RateView rate={asRate(row.parent_strict)} /> : "—"}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
      {cov.missing.length > 0 && (
        <p className="mt-2 text-[11px] leading-snug text-ink-500">
          <span className="font-medium">{cov.missing.join(", ")}</span> produced no
          items. Aggregative questions in particular are the ones no amount of
          reranking fixes, so the headline is silent on the hardest category.
        </p>
      )}
    </Panel>
  );
}

/* --------------------------------------------------- dense / bm25 / rrf table */

function HybridComparison({ h }: { h: NonNullable<InspectorResponse["hybrid_comparison"]> }) {
  return (
    <Panel
      title="Dense · BM25 · RRF"
      subtitle={`same golden set, same budget, one fill rule — rrf_k=${h.rrf_k}`}
    >
      <table className="w-full text-[12px]">
        <thead className="text-[11px] text-ink-400">
          <tr className="text-left">
            <th className="py-1 font-medium">budget</th>
            {h.modes.map((m) => (
              <th key={m} className="font-medium">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {h.budgets.map((b) => (
            <tr key={b} className="border-t border-paper-300">
              <td className="py-1.5 tabular-nums text-ink-600">{b}</td>
              {h.modes.map((m) => {
                const r = asRate(h.results[`${m}@${b}`]?.headline?.child_strict);
                return (
                  <td key={m} className="text-ink-900">
                    <RateView rate={r} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] leading-snug text-ink-500">
        The confidence intervals overlap at every budget: on this sample the RRF
        gain is <span className="font-medium">not distinguishable from noise</span>.
        The measured effect is on exact-identifier queries at tight budgets, where
        dense retrieval loses tokens embeddings compress away.
      </p>
    </Panel>
  );
}

/* ------------------------------------------------------------- judge + defers */

function JudgeGate({ gate, judged }: { gate: InspectorResponse["judge_gate"]; judged: any }) {
  return (
    <Panel
      title="Generation tier — the judge"
      right={<StateBadge state={gate.may_emit_judged_metrics ? "HOLDS" : "NOT_MEASURED"} />}
    >
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-500">
        <span>judge {gate.judge_model}</span>
        <span>generator {gate.generator_model}</span>
        <span>key {gate.detail?.key}</span>
      </div>

      {/* THE VALIDATION, ON THE PAGE. The kappa that gates every number below was
          computed, recorded and then shown nowhere -- so the panel asserted
          "HOLDS" and gave the reader no way to check what it holds against. The
          estimate_type line travels WITH the number on purpose: a
          class-balanced sample proves the judge can separate the classes, not
          that it is precise on a population that is nearly all one class, and a
          kappa quoted without that distinction invites the stronger reading. */}
      {gate.detail?.kappa != null && (
        <div className="mt-2 rounded-md border border-paper-300 bg-paper-100 p-2.5">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[12px] text-ink-900">
            <span className="font-medium">
              Cohen's kappa <span className="tabular-nums">{gate.detail.kappa}</span>
            </span>
            <span className="text-[11px] text-ink-500 tabular-nums">
              raw {gate.detail.raw_agreement} · chance {gate.detail.chance_agreement} ·
              n={gate.detail.n}
            </span>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-ink-400">
            {gate.detail.estimate_type}
          </p>
        </div>
      )}
      {!gate.may_emit_judged_metrics ? (
        <div className="mt-3 rounded-md border border-dashed border-ink-400/50 bg-paper-200 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
            Judged metrics withheld
          </div>
          <p className="mt-1 text-[12px] leading-snug text-ink-600">
            {gate.withheld_because}
          </p>
          <p className="mt-1.5 text-[11px] leading-snug text-ink-400">
            An unvalidated judge is not a weak signal, it is an unknown one — so no
            number is shown at all, rather than a number with a caveat beside it.
          </p>
        </div>
      ) : (
        <div className="mt-3 space-y-2 text-[12px] text-ink-900">
          {/* THE DENOMINATOR, FIRST AND ON ITS OWN LINE.
              This panel used to print "supported 90/91 = 99%" and nothing else.
              17 of those 91 rows were ABSTENTIONS, which the judge scores
              "supported" -- correctly, since an answer asserting nothing asserts
              nothing false. So a fifth of the headline was vacuous passes, and
              the metric improved whenever the system answered FEWER questions.
              A rate whose denominator is invisible invites exactly that misread,
              and it was misread by the person who wrote it. */}
          {judged?.n_answered != null && (
            <div className="flex flex-wrap items-baseline gap-x-2 text-[11px] text-ink-500">
              <span>
                over{" "}
                <span className="font-medium tabular-nums text-ink-900">
                  {judged.n_answered}
                </span>{" "}
                answers
              </span>
              <span className="text-ink-400">
                ({judged.n_abstained} abstained
                {judged.n_starved ? `, ${judged.n_starved} starved by budget` : ""})
              </span>
            </div>
          )}
          <div>
            supported <RateView rate={asRate(judged?.supported)} />
          </div>
          <div>
            answers the question <RateView rate={asRate(judged?.answers_question)} />
          </div>

          {/* Abstention is COVERAGE, not faithfulness, so it is reported beside
              it rather than inside it -- and never averaged in. Declining to
              answer is a real cost even when it is the honest response. */}
          {judged?.abstention_rate && (
            <div className="mt-3 border-t border-paper-300 pt-2">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-ink-500">
                Coverage — questions declined
              </div>
              <div className="mt-1">
                overall <RateView rate={asRate(judged.abstention_rate)} />
              </div>
              {judged.abstention_by_stratum && (
                <table className="mt-1.5 w-full text-[11px]">
                  <tbody>
                    {Object.entries(
                      judged.abstention_by_stratum as Record<string, any>,
                    )
                      .sort((a, b) => (b[1]?.hits ?? 0) - (a[1]?.hits ?? 0))
                      .filter(([, v]) => (v?.hits ?? 0) > 0)
                      .map(([st, v]) => (
                        <tr key={st} className="border-t border-paper-200">
                          <td className="py-1 text-ink-600">{st}</td>
                          <td className="py-1 text-right">
                            <RateView rate={asRate(v)} />
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              )}
              <p className="mt-1.5 text-[11px] leading-snug text-ink-400">
                Every abstention was checked by re-retrieving its evidence: none had
                a complete needle set in the delivered context. The model declines
                when evidence is incomplete, not when it is present — so these are
                retrieval-tier, and table and figure questions are where they
                concentrate.
              </p>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

function Deferrals({ d }: { d: InspectorResponse["deferred"] }) {
  return (
    <Panel title="Deferred decisions" subtitle={d.note}>
      <ul className="divide-y divide-paper-300">
        {d.deferrals.map((x) => (
          <li key={x.name} className="py-2">
            <div className="flex items-center gap-2">
              <span className="text-[12px] font-medium text-ink-900">{x.name}</span>
              <span className="text-[10px] uppercase tracking-wide text-ink-400">
                {x.guide_module}
              </span>
              <StateBadge state={x.expired ? "FAILS" : "HOLDS"} />
              {x.expired && (
                <span className="text-[10px] uppercase tracking-wide text-red-800">
                  precondition changed — revisit
                </span>
              )}
            </div>
            <div className="mt-0.5 text-[12px] text-ink-600">{x.decision}</div>
            <div className="mt-0.5 text-[11px] leading-snug text-ink-500">
              revisit when: {x.revisit_when}
            </div>
            {x.evidence && (
              <div className="mt-0.5 font-mono text-[10px] text-ink-400">{x.evidence}</div>
            )}
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function Regressions({ ev }: { ev: NonNullable<InspectorResponse["eval"]> }) {
  const rows = Object.entries(ev.regression_tests ?? {});
  if (!rows.length) return null;
  return (
    <Panel
      title="Regression tests"
      subtitle="the authored fixture — never averaged into any score"
    >
      <ul className="space-y-1.5">
        {rows.map(([q, v]) => (
          <li key={q} className="flex items-start gap-2 text-[12px]">
            <StateBadge
              state={
                v.passes_at_budget == null
                  ? "FAILS"
                  : v.passes_at_headline
                    ? "HOLDS"
                    : "NOT_MEASURED"
              }
            />
            <span className="min-w-0 text-ink-600">
              {q}
              {v.passes_at_budget != null && !v.passes_at_headline && (
                <span className="ml-1 text-ink-400">
                  passes only at ≥{v.passes_at_budget} tokens
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-[11px] leading-snug text-ink-400">
        A budget-sensitive test is not a failing build. Recording the budget it
        needs keeps “broken” and “needs more room” distinguishable.
      </p>
    </Panel>
  );
}

/* -------------------------------------------------------------------- screen */

export function Inspector() {
  const [d, setD] = useState<InspectorResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.inspector().then(setD).catch((e) => setErr(e.message));
  }, []);

  if (err) return <div className="p-6 text-[13px] text-red-800">{err}</div>;
  if (!d) return <div className="p-6"><Spinner label="reading artifacts" /></div>;

  return (
    <div className="h-full overflow-y-auto px-6 py-5">
      {/* A READING ORDER, roughly: is it trustworthy → how good is it → why.
          Twelve cards of identical size and weight gave a reader who did not
          already know the argument no path through the page. The three that
          carry it -- the headline, the counts that must agree, and the shape of
          the failures -- now run full width and come first; the supporting
          detail pairs off below. */}
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="xl:col-span-2">
          <Headline d={d} />
        </div>
        <div className="xl:col-span-2">
          <Reconciliation data={d.reconciliation} />
        </div>
        {/* The recall curve is the best artifact on this page and was rendering
            at half width with a seven-row table competing beside it -- neither
            winning. Full width, so the knee can be read off the chart. */}
        {d.eval && (
          <div className="xl:col-span-2">
            <BudgetSweep sweep={d.eval.budget_sweep} />
          </div>
        )}
        {d.failures && (
          <div className="xl:col-span-2">
            <Failures f={d.failures} />
          </div>
        )}
        {d.eval && <Strata ev={d.eval} />}
        <JudgeGate gate={d.judge_gate} judged={d.judged} />
        {d.hybrid_comparison && <HybridComparison h={d.hybrid_comparison} />}
        {d.eval && <Regressions ev={d.eval} />}
        <div className="xl:col-span-2">
          <Deferrals d={d.deferred} />
        </div>
      </div>
    </div>
  );
}
