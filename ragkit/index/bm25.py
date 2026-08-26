"""
BM25, written by hand. Guide module M2.

WHY BY HAND: "explain BM25" should be something implemented rather than
memorised. It is about eighty lines, and the two parameters are the whole idea.

WHY AT ALL -- the dense failure mode, stated precisely:

Embeddings compress meaning, and compression destroys exact tokens. `ERR_5521`,
`PN-8891-A`, `efConstruction`, `Recall@10`, `HYBRID_RRF` -- error codes, part
numbers, hyperparameter names, version strings -- all embed to roughly "technical
identifier shaped thing" and stop being distinguishable from each other. BM25
does not compress: a term either occurs or it does not.

The eval says this is the right next layer. `source_hit = 100%` at every budget
(the correct DOCUMENT always surfaces) while `child_strict = 85%` at a realistic
budget -- so every failure is ranking-within-document, which is exactly what a
lexical leg reorders. It is also what a better embedding model does NOT fix,
which is the work the eval already saved.

---------------------------------------------------------------------------
THE TWO PARAMETERS, AND WHAT THEY DO

k1 -- TERM FREQUENCY SATURATION. A term's contribution is tf/(tf + k1*norm), so
      the tenth occurrence of a word adds far less than the second. Without
      saturation, a document that repeats a word forty times outranks one that
      uses it twice and actually answers the question. Set k1=0 and every term
      counts once regardless of frequency; raise it and frequency matters more.

b  -- DOCUMENT LENGTH NORMALISATION, via norm = (1 - b) + b * len/avglen.
      b=0 disables it, so long documents win purely by containing more words.
      b=1 fully normalises. 0.75 is the conventional compromise.

`b` matters unusually much here, and it is a self-inflicted problem: contextual
breadcrumbs are a FIXED cost per child, and breadcrumb length correlates with
section depth. Measured on this corpus, prefix overhead is 8% of body tokens on a
paper and 35% on manual.docx -- a four-fold difference in the same corpus. So
length normalisation is partly normalising an artifact of our own enrichment.
That is worth knowing before tuning b and concluding something about the corpus.

---------------------------------------------------------------------------
THE TOKENIZER IS THE WHOLE EXPERIMENT

A naive `[a-z0-9]+` split turns `ERR_5521` into `err` + `5521`. Both tokens
survive, so it half-works -- and it destroys the thing that made the query
answerable: `err` occurs in hundreds of chunks and carries almost no IDF, while
`ERR_5521` occurs in one and carries a lot.

So identifiers are emitted WHOLE **and** split into parts:

    ERR_5521      -> ["err_5521", "err", "5521"]
    PN-8891-A     -> ["pn-8891-a", "pn", "8891", "a"]
    Recall@10     -> ["recall@10", "recall", "10"]
    efConstruction-> ["efconstruction", "ef", "construction"]

An exact query then matches the high-IDF whole token, and a loose query ("error
5521", "recall at 10") still matches the parts. Emitting only the whole token
would break loose queries; emitting only parts is the naive failure. Both costs
index size, which at this scale is free.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import config
from ..ingest.document import Chunk, ChunkRole

# Runs of alphanumerics joined by internal ._-@/# -- the shape of an identifier.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._\-@/#][A-Za-z0-9]+)*")
_SPLIT = re.compile(r"[._\-@/#]")
# camelCase / PascalCase boundary, so efConstruction also yields ef + construction.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Identifiers whole, plus their parts. See the module docstring."""
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        out.append(low)
        parts = [p for p in _SPLIT.split(low) if p]
        if len(parts) > 1:
            out.extend(parts)
        for part in parts:
            camel = [c.lower() for c in _CAMEL.split(raw) if c]
            if len(camel) > 1:
                out.extend(camel)
            break
    return out


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float
    rank: int
    matched: list[str]


class BM25Index:
    """Inverted index over child chunks.

    Deliberately not `rank_bm25`: that library is used in the test below as an
    independent implementation to check this one against. Two implementations
    that agree is evidence; one implementation is an assertion.
    """

    def __init__(
        self,
        children: Sequence[Chunk],
        *,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.children = list(children)

        self.doc_tokens: list[list[str]] = [tokenize(c.embed_text or c.display_text) for c in self.children]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

        self.tf: list[Counter[str]] = [Counter(t) for t in self.doc_tokens]
        self.postings: dict[str, list[int]] = {}
        for i, counts in enumerate(self.tf):
            for term in counts:
                self.postings.setdefault(term, []).append(i)

        n = len(self.children)
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, and the max(0, .)
        # guard: a term in more than half the corpus otherwise scores NEGATIVE,
        # so a document containing a stopword would rank BELOW one that does not.
        self.idf: dict[str, float] = {}
        for term, docs in self.postings.items():
            df = len(docs)
            self.idf[term] = max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    # -- search --------------------------------------------------------------

    def _score_doc(self, i: int, q_terms: Sequence[str]) -> tuple[float, list[str]]:
        norm = (1.0 - self.b) + self.b * (self.doc_len[i] / self.avgdl if self.avgdl else 1.0)
        total = 0.0
        matched: list[str] = []
        counts = self.tf[i]
        for term in q_terms:
            f = counts.get(term, 0)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue
            total += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * norm)
            matched.append(term)
        return total, matched

    def search_k(self, query: str, k: int = 10) -> list[BM25Hit]:
        q_terms = tokenize(query)
        # Only documents that share at least one query term can score, so the
        # postings list bounds the work -- that is the whole point of an inverted
        # index, and it is why BM25 costs ~10ms where a cross-encoder costs 300.
        candidates: set[int] = set()
        for t in set(q_terms):
            candidates.update(self.postings.get(t, ()))
        scored = []
        for i in candidates:
            s, matched = self._score_doc(i, q_terms)
            if s > 0:
                scored.append((s, i, matched))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            BM25Hit(self.children[i], float(s), r, m)
            for r, (s, i, m) in enumerate(scored[:k])
        ]

    def search_budget(
        self, query: str, *, token_budget: int, max_items: int = 200
    ) -> list[BM25Hit]:
        """Same strict fill rule as the dense index, for the same reason.

        Never exceeds the budget, even if that returns nothing. A unit allowed to
        overshoot is simply given more text, which destroys any comparison
        between units -- measured once already, as a fake crossover in the sweep.
        """
        from ..gemini import count_tokens

        hits = self.search_k(query, max_items)
        out: list[BM25Hit] = []
        spent = 0
        for h in hits:
            cost = count_tokens(h.chunk.embed_text or h.chunk.display_text)
            if spent + cost > token_budget:
                break
            spent += cost
            out.append(h)
        return out

    # -- persistence ---------------------------------------------------------
    #
    # The index is rebuilt from chunks rather than serialised: it is a few
    # hundred milliseconds over 800 chunks, and a persisted inverted index is one
    # more artifact that can silently disagree with the corpus it describes.
    # Two representations that must agree is the failure mode this project keeps
    # finding; not creating the second representation avoids it entirely.

    # `from_index_dir` was removed: HybridIndex builds BM25Index(dense.children)
    # directly, so the loader-from-disk path had no caller and was a second way to
    # construct the same object -- two constructors that must agree is the shape
    # of bug this project keeps finding. Deleted rather than left as a trap.

    def stats(self) -> dict[str, Any]:
        singletons = sum(1 for t, docs in self.postings.items() if len(docs) == 1)
        return {
            "k1": self.k1,
            "b": self.b,
            "n_docs": len(self.children),
            "vocab": len(self.postings),
            "avgdl": round(self.avgdl, 1),
            # A high-IDF vocabulary is the point: terms occurring in exactly one
            # chunk are the ones dense retrieval cannot distinguish.
            "singleton_terms": singletons,
            "singleton_share": round(singletons / max(len(self.postings), 1), 3),
        }
