"""
SPORE AI Framework — Tokenization Layer (§2.2.2)

Provides deterministic subword-level tokenisation using a BPE-inspired
byte-pair approach.  For production deployments a pretrained vocabulary
(e.g. GPT-2 or SentencePiece) should be substituted; this implementation
satisfies the specification requirements without heavy dependencies.

Requirements met:
- Subword tokenisation (BPE-style merge rules, §2.2.2)
- Deterministic encoding (same input → same token sequence)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


class SporeTokenizer:
    """
    Lightweight subword tokeniser (BPE-inspired, no external dependencies).

    Encoding pipeline:
        1. Unicode normalisation + lower-casing (optional)
        2. Whitespace + punctuation pre-tokenisation
        3. Character-level initialisation
        4. Greedy merge of the most-frequent byte-pairs (BPE rule application)
        5. Encode each sub-word to an integer id

    The vocabulary is built incrementally via ``train()``; a minimal default
    vocabulary covering ASCII printable characters is always present.
    """

    # Internal sentinel for word boundaries (never appears in real text)
    _EOW = "</w>"

    def __init__(self, vocab_size: int = 8192, lowercase: bool = True) -> None:
        self.vocab_size = vocab_size
        self.lowercase  = lowercase

        # token → id (deterministic)
        self._vocab:    Dict[str, int] = {}
        # id → token
        self._id_vocab: Dict[int, str] = {}
        # BPE merge rules: (left, right) → merged
        self._merges:   List[Tuple[str, str]] = []

        self._init_base_vocab()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, texts: List[str], num_merges: int = 1000) -> None:
        """
        Learn BPE merge rules from a corpus of texts.

        Args:
            texts: Raw text strings used to compute pair frequencies.
            num_merges: Number of BPE merge operations to learn.
        """
        # Build initial word frequency table
        word_freq: Dict[str, int] = {}
        for text in texts:
            for word in self._pretokenise(text):
                chars = " ".join(list(word)) + " " + self._EOW
                word_freq[chars] = word_freq.get(chars, 0) + 1

        for _ in range(num_merges):
            if len(self._vocab) >= self.vocab_size:
                break
            pairs = self._count_pairs(word_freq)
            if not pairs:
                break
            best = max(pairs, key=lambda p: pairs[p])
            word_freq = self._merge_pair(best, word_freq)
            self._merges.append(best)
            merged = best[0] + best[1]
            if merged not in self._vocab:
                new_id = len(self._vocab)
                self._vocab[merged]    = new_id
                self._id_vocab[new_id] = merged

    def encode(self, text: str) -> List[int]:
        """Return a list of token ids for the input text."""
        tokens = self._tokenise(text)
        unk = self._vocab.get("<unk>", 0)
        return [self._vocab.get(t, unk) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from a list of token ids (best-effort)."""
        tokens = [self._id_vocab.get(i, "<unk>") for i in ids]
        text   = "".join(t.replace(self._EOW, " ") for t in tokens)
        return text.strip()

    def tokenise(self, text: str) -> List[str]:
        """Return token strings (not ids) for the input text."""
        return self._tokenise(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_base_vocab(self) -> None:
        """Initialise a minimal ASCII + special-token vocabulary."""
        specials = ["<pad>", "<unk>", "<bos>", "<eos>", self._EOW]
        for token in specials:
            if token not in self._vocab:
                idx = len(self._vocab)
                self._vocab[token]    = idx
                self._id_vocab[idx]   = token

        # Single printable ASCII characters
        for c in (chr(i) for i in range(32, 127)):
            if c not in self._vocab:
                idx = len(self._vocab)
                self._vocab[c]    = idx
                self._id_vocab[idx] = c

    def _pretokenise(self, text: str) -> List[str]:
        """Split text into words before BPE is applied."""
        if self.lowercase:
            text = text.lower()
        # Split on whitespace and isolate punctuation as separate tokens
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [t for t in tokens if t.strip()]

    def _tokenise(self, text: str) -> List[str]:
        """Apply learned BPE merge rules to produce subword tokens."""
        result: List[str] = []
        for word in self._pretokenise(text):
            chars = list(word) + [self._EOW]
            for left, right in self._merges:
                i = 0
                merged: List[str] = []
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == left and chars[i + 1] == right:
                        merged.append(left + right)
                        i += 2
                    else:
                        merged.append(chars[i])
                        i += 1
                chars = merged
            result.extend(chars)
        return result

    @staticmethod
    def _count_pairs(word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs: Dict[Tuple[str, str], int] = {}
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    @staticmethod
    def _merge_pair(
        pair: Tuple[str, str],
        word_freq: Dict[str, int],
    ) -> Dict[str, int]:
        left, right = pair
        bigram      = re.escape(left + " " + right)
        pattern     = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_freq: Dict[str, int] = {}
        for word, freq in word_freq.items():
            new_word          = pattern.sub(left + right, word)
            new_freq[new_word] = new_freq.get(new_word, 0) + freq
        return new_freq
