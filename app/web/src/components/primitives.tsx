/**
 * Shared primitives. Each one encodes a rule the backend already enforces, so
 * the UI cannot present a number more confidently than the data supports.
 */
import type { ReactNode } from "react";
import type { Check, Rate } from "../api";

/**
 * A measured rate. Renders "2 of 3" when the sample is below the floor and a
 * percentage only when it is not.
 *
 * The backend sets `sufficient`; this component just refuses to override it. A
 * percentage implies a precision that three observations do not have, and it is
 * exactly the kind of number that gets quoted later without its sample size --
 * which is how an 18% figure got published and then withdrawn during this build.
 */
export function RateView({ rate, className = "" }: { rate?: Rate; className?: string }) {
  if (!rate) return <span className="text-ink-400">—</span>;
  return (
    <span className={className}>
      <span className="tabular-nums">{rate.label}</span>
      {rate.sufficient && rate.ci95 && (
        <span className="ml-1.5 text-[11px] text-ink-400 tabular-nums">
          ±{Math.round(((rate.ci95[1] - rate.ci95[0]) / 2) * 100)}pt
        </span>
      )}
      {!rate.sufficient && rate.n > 0 && (
        <span
          className="ml-1.5 rounded border border-ink-400/30 px-1 text-[10px] uppercase tracking-wide text-ink-400"
          title="Below the sample-size floor: reported as a count, not a rate."
        >
          counts only
        </span>
      )}
    </span>
  );
}

/**
 * HOLDS / FAILS / NOT_MEASURED.
 *
 * NOT_MEASURED is a first-class state and deliberately not styled like a pass.
 * An absent value invites investigation; one dressed as success ends it.
 */
export function StateBadge({ state }: { state: Check["state"] | string }) {
  const map: Record<string, string> = {
    HOLDS: "border-quote-600/30 bg-quote-600/5 text-quote-600",
    FAILS: "border-red-700/30 bg-red-700/5 text-red-800",
    NOT_MEASURED: "border-dashed border-ink-400/50 bg-paper-200 text-ink-500",
  };
  const label = state === "NOT_MEASURED" ? "not measured" : state.toLowerCase();
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
        map[state] ?? map.NOT_MEASURED
      }`}
    >
      {label}
    </span>
  );
}

/** The pipeline fingerprint. Every figure on a page inherits it. */
export function FingerprintChip({
  parts,
  hash,
}: {
  parts: string[];
  hash?: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-500">
      {parts.map((p) => (
        <span key={p} className="rounded bg-paper-200 px-1.5 py-0.5 font-mono">
          {p}
        </span>
      ))}
      {hash && (
        <span className="rounded bg-paper-300 px-1.5 py-0.5 font-mono text-ink-600">
          {hash}
        </span>
      )}
    </div>
  );
}

export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-lg border border-paper-400 bg-paper-50 ${className}`}
    >
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-paper-300 px-4 py-3">
          <div>
            {title && (
              <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-500">
                {title}
              </h2>
            )}
            {subtitle && (
              <div className="mt-1 text-[12px] leading-snug text-ink-500">{subtitle}</div>
            )}
          </div>
          {right}
        </header>
      )}
      <div className="px-4 py-3">{children}</div>
    </section>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-[12px] text-ink-500">
      <span className="size-3 animate-spin rounded-full border-2 border-ink-400/40 border-t-ink-600" />
      {label}
    </span>
  );
}

/** A degradation notice. Impact, fallback and remedy — never just "warning". */
export function DegradationBanner({ items }: { items: any[] }) {
  if (!items?.length) return null;
  return (
    <div className="rounded-md border border-chart-600/40 bg-chart-600/5 px-3 py-2.5">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-chart-600">
        Degraded — this run did not do everything it was asked to
      </div>
      <ul className="mt-2 space-y-2">
        {items.map((d, i) => (
          <li key={i} className="text-[12px] leading-snug text-ink-600">
            <span className="font-medium">
              {d.count && d.attempted ? `${d.count} of ${d.attempted}: ` : ""}
              {d.impact}
            </span>
            <div className="text-ink-500">instead: {d.fallback}</div>
            <div className="text-ink-500">you can: {d.remedy}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
