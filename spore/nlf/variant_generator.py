"""
NLF Variant Generator
Generates surface variants from base questions (framework §3).

Each base question yields 100+ surface variants through six techniques:
  1. Lexical substitution  – synonym replacement
  2. Syntactic alternation – structural paraphrase
  3. Pragmatic modulation  – formality/register shift
  4. Ellipsis expansion    – full vs. reduced forms
  5. Contextual embedding  – surrounding phrase attachment
  6. Noise tolerance       – typo/variant spellings
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from .models import Register, SurfaceVariant, VariantTechnique


# ---------------------------------------------------------------------------
# Simple bag-of-words cosine similarity (no external dependencies)
# ---------------------------------------------------------------------------

def _token_vector(text: str) -> dict[str, int]:
    tokens = re.findall(r"[a-z']+", text.lower())
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine_similarity(a: str, b: str) -> float:
    va, vb = _token_vector(a), _token_vector(b)
    if not va or not vb:
        return 0.0
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in va)
    mag_a = math.sqrt(sum(v * v for v in va.values()))
    mag_b = math.sqrt(sum(v * v for v in vb.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _max_similarity(candidate: str, existing: Sequence[str]) -> float:
    if not existing:
        return 0.0
    return max(_cosine_similarity(candidate, e) for e in existing)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_DEDUP_THRESHOLD = 0.85


def validate_variant(
    candidate: str,
    existing_texts: Sequence[str],
    *,
    register: Register,
    technique: VariantTechnique,
) -> SurfaceVariant:
    """
    Validate a candidate variant and return a SurfaceVariant.

    Raises ValueError if:
    - The candidate is empty or whitespace-only.
    - Cosine similarity to any existing variant is ≥ 0.85 (§3.3).
    """
    text = candidate.strip()
    if not text:
        raise ValueError("Variant text must not be empty.")

    sim = _max_similarity(text, existing_texts)
    if sim >= _DEDUP_THRESHOLD:
        raise ValueError(
            f"Variant '{text}' too similar to existing variants (sim={sim:.3f} ≥ {_DEDUP_THRESHOLD})."
        )

    return SurfaceVariant(
        text=text,
        technique=technique,
        register=register,
        validated=True,
        similarity_score=sim,
    )


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------

class VariantGenerator:
    """
    Generates surface variants for a base question using all six NLF techniques.

    Usage::

        gen = VariantGenerator()
        variants = gen.generate(
            base="How are you?",
            templates={
                VariantTechnique.LEXICAL_SUBSTITUTION: [
                    ("How are you?", Register.NEUTRAL),
                    ("How are you doing?", Register.NEUTRAL),
                ],
                VariantTechnique.PRAGMATIC_MODULATION: [
                    ("What's up?", Register.CASUAL),
                    ("How do you do?", Register.FORMAL),
                ],
                ...
            }
        )
    """

    def __init__(self, dedup_threshold: float = _DEDUP_THRESHOLD) -> None:
        self.dedup_threshold = dedup_threshold

    def generate(
        self,
        base: str,
        templates: dict[VariantTechnique, list[tuple[str, Register]]],
    ) -> list[SurfaceVariant]:
        """
        Generate and deduplicate variants from technique-keyed template lists.

        Args:
            base: Canonical form of the base question (always included first).
            templates: Mapping from technique → list of (text, register) pairs.

        Returns:
            List of validated SurfaceVariant objects with no pair exceeding
            the deduplication threshold.
        """
        accepted: list[SurfaceVariant] = []
        seen_texts: list[str] = []

        # Always include the canonical form as a SYNTACTIC_ALTERNATION neutral variant
        canon_variant = SurfaceVariant(
            text=base.strip(),
            technique=VariantTechnique.SYNTACTIC_ALTERNATION,
            register=Register.NEUTRAL,
            validated=True,
            similarity_score=0.0,
        )
        accepted.append(canon_variant)
        seen_texts.append(base.strip())

        for technique, pairs in templates.items():
            for text, register in pairs:
                text = text.strip()
                if not text:
                    continue
                sim = _max_similarity(text, seen_texts)
                if sim >= self.dedup_threshold:
                    continue  # Skip near-duplicate
                variant = SurfaceVariant(
                    text=text,
                    technique=technique,
                    register=register,
                    validated=True,
                    similarity_score=sim,
                )
                accepted.append(variant)
                seen_texts.append(text)

        return accepted

    # ------------------------------------------------------------------
    # Convenience method: load from a simple dict spec
    # ------------------------------------------------------------------

    @staticmethod
    def from_spec(spec: dict) -> list[SurfaceVariant]:
        """
        Build variants from a JSON-serialisable spec dict.

        Spec format::

            {
              "base": "How are you?",
              "variants": [
                {
                  "text": "How's it going?",
                  "technique": "syntactic_alternation",
                  "register": "casual"
                },
                ...
              ]
            }
        """
        base = spec.get("base", "").strip()
        raw_variants = spec.get("variants", [])

        accepted: list[SurfaceVariant] = []
        seen: list[str] = [base] if base else []

        if base:
            accepted.append(SurfaceVariant(
                text=base,
                technique=VariantTechnique.SYNTACTIC_ALTERNATION,
                register=Register.NEUTRAL,
                validated=True,
                similarity_score=0.0,
            ))

        for item in raw_variants:
            text = item.get("text", "").strip()
            if not text:
                continue
            try:
                technique = VariantTechnique(item.get("technique", "syntactic_alternation"))
                register = Register(item.get("register", "neutral"))
            except ValueError:
                continue
            sim = _max_similarity(text, seen)
            if sim >= _DEDUP_THRESHOLD:
                continue
            accepted.append(SurfaceVariant(
                text=text,
                technique=technique,
                register=register,
                validated=True,
                similarity_score=sim,
            ))
            seen.append(text)

        return accepted
