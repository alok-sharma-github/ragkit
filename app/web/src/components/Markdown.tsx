/**
 * A small markdown renderer for answer text.
 *
 * WHY NOT react-markdown. Two reasons, and the second is the real one:
 *
 *   scope   the answer schema produces a known, narrow subset -- headings,
 *           bold, italic, inline code, bullet and numbered lists, paragraphs.
 *           A general CommonMark + GFM implementation is a large dependency for
 *           a grammar we control the producer of.
 *   safety  every renderer that takes a shortcut reaches for
 *           dangerouslySetInnerHTML, and this text is MODEL OUTPUT QUOTING USER
 *           DOCUMENTS -- two untrusted layers stacked. The corpus already
 *           contains raw HTML that survives the parser (`<u>HOME SCREEN:</u>`
 *           appears in real breadcrumbs), so HTML in this string is not
 *           hypothetical, it is present today. Rendering to React text nodes
 *           makes injection structurally impossible rather than filtered-against:
 *           a stray tag is displayed as characters, which is both safe and
 *           honest about what the document contains.
 *
 * This is NOT the primary answer surface. A grounded answer renders as a list of
 * claims, each carrying its own citation chips, because a claim is the unit that
 * can be verified and a paragraph is not. This renderer is for the case where
 * there are no claims to show -- an ungrounded answer, or one whose JSON was cut
 * off at the output limit and whose prose was salvaged without its citations.
 */
import React from "react";

/** `**bold**`, `*italic*`, `_italic_`, `` `code` `` -> React nodes. */
function inline(text: string, keyBase: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  // One pass, longest-delimiter-first so `**` is never mistaken for two `*`.
  const re = /(\*\*[^*]+\*\*|`[^`]+`|(?<![A-Za-z0-9])_[^_]+_(?![A-Za-z0-9])|\*[^*]+\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let n = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const k = `${keyBase}-i${n++}`;
    if (tok.startsWith("**")) {
      out.push(
        <strong key={k} className="font-semibold text-ink-900">
          {tok.slice(2, -2)}
        </strong>,
      );
    } else if (tok.startsWith("`")) {
      out.push(
        <code
          key={k}
          className="rounded bg-paper-200 px-1 py-0.5 font-mono text-[12px] text-ink-900"
        >
          {tok.slice(1, -1)}
        </code>,
      );
    } else {
      out.push(
        <em key={k} className="italic">
          {tok.slice(1, -1)}
        </em>,
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Inline-only formatting, for text that must stay on one line (claim text). */
export function Inline({ text }: { text: string }) {
  return <>{inline(text, "x")}</>;
}

type Block =
  | { kind: "h"; level: number; text: string }
  | { kind: "p"; text: string }
  | { kind: "ul" | "ol"; items: string[] }
  | { kind: "pre"; text: string };

function parse(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;

  const flushPara = (buf: string[]) => {
    if (buf.length) blocks.push({ kind: "p", text: buf.join(" ").trim() });
    buf.length = 0;
  };
  const para: string[] = [];

  while (i < lines.length) {
    const line = lines[i];

    // A fence with no closer is normal here: the text may have been truncated
    // mid-block, so run to the end rather than dropping the remainder.
    if (/^\s*```/.test(line)) {
      flushPara(para);
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++;
      blocks.push({ kind: "pre", text: body.join("\n") });
      continue;
    }

    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flushPara(para);
      blocks.push({ kind: "h", level: h[1].length, text: h[2].trim() });
      i++;
      continue;
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const num = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (bullet || num) {
      flushPara(para);
      const kind = bullet ? "ul" : "ol";
      const items: string[] = [];
      while (i < lines.length) {
        const b = /^\s*[-*+]\s+(.*)$/.exec(lines[i]);
        const o = /^\s*\d+[.)]\s+(.*)$/.exec(lines[i]);
        const mm = kind === "ul" ? b : o;
        if (mm) {
          items.push(mm[1].trim());
          i++;
        } else if (/^\s+\S/.test(lines[i]) && items.length) {
          // continuation line of the previous item
          items[items.length - 1] += " " + lines[i].trim();
          i++;
        } else break;
      }
      blocks.push({ kind, items });
      continue;
    }

    if (!line.trim()) {
      flushPara(para);
      i++;
      continue;
    }

    para.push(line.trim());
    i++;
  }
  flushPara(para);
  return blocks;
}

const H = ["text-[17px]", "text-[15px]", "text-[14px]", "text-[13px]", "text-[13px]", "text-[13px]"];

export default function Markdown({ text }: { text: string }) {
  if (!text?.trim()) return null;
  const blocks = parse(text);
  return (
    <div className="mt-3 font-serif text-[15px] leading-relaxed text-ink-900">
      {blocks.map((b, i) => {
        const k = `b${i}`;
        if (b.kind === "h")
          return (
            <div
              key={k}
              className={`mt-4 first:mt-0 font-sans font-semibold tracking-tight text-ink-900 ${
                H[b.level - 1]
              }`}
            >
              {inline(b.text, k)}
            </div>
          );
        if (b.kind === "pre")
          return (
            <pre
              key={k}
              className="mt-2 overflow-x-auto rounded border border-paper-300 bg-paper-100 p-3 font-mono text-[12px] leading-relaxed text-ink-900"
            >
              {b.text}
            </pre>
          );
        if (b.kind === "ul" || b.kind === "ol") {
          const Tag = b.kind === "ul" ? "ul" : "ol";
          return (
            <Tag
              key={k}
              className={`mt-2 space-y-1 pl-5 ${
                b.kind === "ul" ? "list-disc" : "list-decimal"
              } marker:text-ink-400`}
            >
              {b.items.map((it, j) => (
                <li key={`${k}-${j}`}>{inline(it, `${k}-${j}`)}</li>
              ))}
            </Tag>
          );
        }
        // Explicit rather than a fallthrough. TS would not narrow the union
        // down to "p" here -- the JSX `<Tag>` above with a string-union tag
        // makes that branch's return opaque to the checker -- so the last case
        // is stated, and an unhandled kind renders nothing instead of throwing.
        if (b.kind === "p")
          return (
            <p key={k} className="mt-2 first:mt-0">
              {inline(b.text, k)}
            </p>
          );
        return null;
      })}
    </div>
  );
}
