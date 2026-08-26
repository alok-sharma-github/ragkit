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
        <div className="mt-3 space-y-1 text-[12px] text-ink-900">
          <div>
            supported <RateView rate={asRate(judged?.supported)} />
          </div>
          <div>
            answers the question <RateView rate={asRate(judged?.answers_question)} />
          </div>
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
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 xl:grid-cols-2">
        <div className="space-y-4 xl:col-span-2">
          <Reconciliation data={d.reconciliation} />
        </div>
        {d.eval && <BudgetSweep sweep={d.eval.budget_sweep} />}
        {d.eval && <Strata ev={d.eval} />}
        {d.hybrid_comparison && <HybridComparison h={d.hybrid_comparison} />}
        <JudgeGate gate={d.judge_gate} judged={d.judged} />
        {d.eval && <Regressions ev={d.eval} />}
        <Deferrals d={d.deferred} />
      </div>
    </div>
  );
}
