"""
corpus.py — text corpus utilities for activation collection.

Two loading strategies:
  1. CorpusLoader.diverse_sentences(n)
       A curated multi-domain sentence list built into this file.
       No external downloads required.  Good for quick local runs.

  2. CorpusLoader.from_openwebtext(n, seed)
       Streams a subset of the Hugging Face openwebtext dataset.
       Requires:  pip install datasets

Each returned sentence is a single coherent clause (≤ 100 tokens) so
that mean-pooled residual-stream representations carry clear semantic signal.
"""

from __future__ import annotations

import logging
import random
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in diverse corpus (~750 base sentences across 14 domains)
# ---------------------------------------------------------------------------

_SENTENCES: List[str] = [
    # ── Science & Nature ────────────────────────────────────────────────────
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "Black holes warp spacetime so severely that not even light can escape.",
    "DNA replication occurs in the nucleus before a cell divides.",
    "The mitochondria generate most of the cell's supply of ATP.",
    "Continental drift is driven by convection currents in the mantle.",
    "Quantum entanglement allows particles to share states across any distance.",
    "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
    "Neurons communicate through electrochemical signals called action potentials.",
    "Enzyme activity is sensitive to changes in temperature and pH.",
    "The ozone layer absorbs most of the Sun's ultraviolet radiation.",
    "Tectonic plates move at roughly the same rate as human fingernails grow.",
    "Water expands when it freezes, which is why ice floats on liquid water.",
    "The human genome contains approximately 3 billion base pairs.",
    "Stars form when a cloud of gas and dust collapses under gravity.",
    "Evolution proceeds through the differential reproductive success of organisms.",
    "Atmospheric pressure decreases with altitude above sea level.",
    "The half-life of carbon-14 is approximately 5,730 years.",
    "Viruses hijack host cell machinery to replicate their genetic material.",
    "The greenhouse effect traps heat in Earth's atmosphere via infrared absorption.",
    "Magnetic field lines emerge from the north pole and enter the south pole.",
    "Sound travels faster through liquids than through gases.",
    "Bioluminescence in deep-sea creatures is produced by chemical reactions.",
    "Coral reefs are among the most biodiverse ecosystems on the planet.",
    "Supernovas synthesise heavy elements that scatter across the galaxy.",
    "The appendix may play a role in gut microbiome recovery after illness.",

    # ── Technology & Computing ───────────────────────────────────────────────
    "A neural network learns by adjusting weights to minimise a loss function.",
    "Gradient descent iteratively moves parameters in the direction of steepest descent.",
    "Transformers use self-attention to model relationships between all token pairs.",
    "The softmax function converts raw logits into a probability distribution.",
    "Overfitting occurs when a model memorises training data instead of generalising.",
    "Regularisation techniques such as dropout reduce overfitting in deep networks.",
    "Convolutional layers detect local patterns in images through shared filters.",
    "Recurrent networks struggle with long-range dependencies due to vanishing gradients.",
    "Tokenisation splits text into sub-word units before feeding it to a language model.",
    "The attention mechanism allows the model to focus on relevant parts of the input.",
    "Backpropagation computes gradients using the chain rule of calculus.",
    "Version control systems track changes to code over time.",
    "A Turing machine is a theoretical model of computation.",
    "Public-key cryptography relies on the mathematical difficulty of factoring large primes.",
    "Containerisation packages applications with their dependencies for portability.",
    "Cache invalidation is widely considered one of the hardest problems in computing.",
    "Binary search finds an element in a sorted list in O(log n) time.",
    "A relational database organises data into tables linked by keys.",
    "The halting problem is undecidable for general-purpose computing machines.",
    "Floating-point arithmetic introduces small rounding errors in numeric computations.",
    "GPUs parallelize matrix operations, making them well-suited for deep learning.",
    "An API defines a contract between software components for communication.",
    "Distributed systems must handle network partitions, latency, and node failures.",
    "Garbage collection automatically reclaims memory no longer referenced by the program.",
    "Compilers translate high-level source code into machine instructions.",

    # ── Language & Linguistics ───────────────────────────────────────────────
    "Syntax refers to the rules that govern sentence structure in a language.",
    "Semantics is the study of meaning in language.",
    "Pragmatics examines how context influences the interpretation of utterances.",
    "Code-switching occurs when a speaker alternates between two languages in conversation.",
    "The Sapir-Whorf hypothesis suggests language shapes the way people think.",
    "Morphology studies the internal structure of words and their smallest meaningful units.",
    "Prosody encompasses the rhythm, stress, and intonation patterns of speech.",
    "Lexical ambiguity arises when a single word has multiple distinct meanings.",
    "Metaphor maps conceptual structure from a source domain onto a target domain.",
    "Sociolinguistics examines how social factors influence language variation.",
    "Children acquire their first language without explicit instruction.",
    "Sign languages are full natural languages with their own grammars.",
    "A phoneme is the smallest unit of sound that distinguishes meaning in a language.",
    "Computational linguistics applies statistical and rule-based methods to language.",
    "Discourse coherence depends on local cohesion and global topic continuity.",

    # ── Mathematics ─────────────────────────────────────────────────────────
    "The Pythagorean theorem states that the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "Prime numbers have exactly two divisors: one and themselves.",
    "Euler's identity relates five fundamental constants in a single equation.",
    "A group is a set equipped with an associative binary operation, an identity, and inverses.",
    "Proof by induction establishes a statement for all natural numbers.",
    "Fourier transforms decompose a function into its constituent frequencies.",
    "The central limit theorem underpins much of inferential statistics.",
    "A continuous function on a closed interval attains its maximum and minimum.",
    "Probability is a measure that assigns a number between zero and one to events.",
    "Eigenvalues describe how a linear transformation scales specific directions.",
    "Calculus was developed independently by Newton and Leibniz.",
    "The number pi is irrational and transcendental.",
    "A manifold is a topological space that locally resembles Euclidean space.",
    "Graph theory studies networks of vertices connected by edges.",
    "Set theory provides the foundation for most of modern mathematics.",

    # ── History & Society ────────────────────────────────────────────────────
    "The Industrial Revolution transformed manufacturing from hand production to machinery.",
    "The printing press enabled the rapid dissemination of ideas across Europe.",
    "Democracy originated in ancient Athens as a system of direct citizen governance.",
    "The Cold War was a period of geopolitical tension between the US and Soviet Union.",
    "The Renaissance marked a revival of classical art, literature, and learning.",
    "Colonialism had lasting economic and cultural impacts on colonised nations.",
    "The French Revolution challenged the traditional authority of monarchy and church.",
    "World War II resulted in the deaths of an estimated 70 to 85 million people.",
    "The Silk Road connected trade networks across Asia, the Middle East, and Europe.",
    "Nationalism became a powerful political force in nineteenth-century Europe.",
    "The abolition of slavery was a landmark moral and political achievement.",
    "Urbanisation accelerated as people migrated from rural areas to cities.",
    "The United Nations was founded in 1945 to maintain international peace.",
    "Economic inequality persists across generations through inheritance and education.",
    "Propaganda shapes public opinion through selective presentation of information.",

    # ── Philosophy & Cognition ───────────────────────────────────────────────
    "Epistemology is the branch of philosophy concerned with the nature of knowledge.",
    "The mind-body problem asks how mental states relate to physical brain states.",
    "Free will debates centre on whether human actions are causally determined.",
    "Consciousness remains one of the deepest unsolved problems in science.",
    "Heuristics are mental shortcuts that speed up decision-making at some cost to accuracy.",
    "Cognitive biases systematically distort human judgement and perception.",
    "Working memory holds a limited amount of information available for active processing.",
    "Embodied cognition suggests that thinking is shaped by the body and environment.",
    "The frame problem concerns how agents decide which facts remain unchanged after an action.",
    "Intuition and deliberative reasoning operate through different cognitive systems.",
    "Moral philosophy asks what makes actions right or wrong.",
    "Phenomenology focuses on the structure of subjective, first-person experience.",
    "The trolley problem is a thought experiment in ethics about sacrificing one to save many.",
    "Falsifiability is Popper's criterion for demarcating science from non-science.",
    "Social constructivism holds that knowledge is shaped by social and cultural factors.",

    # ── Arts & Literature ────────────────────────────────────────────────────
    "Narrative tension is created through the conflict between character desire and obstacle.",
    "Rhythm and rhyme can reinforce or counterpoint the meaning of a poem.",
    "The unreliable narrator invites readers to question the truth of the story.",
    "Abstract expressionism prioritises emotional intensity over representational accuracy.",
    "Counterpoint in music involves two or more independent melodic lines heard simultaneously.",
    "The hero's journey is a narrative archetype identified by Joseph Campbell.",
    "Stream of consciousness captures the unfiltered flow of a character's inner thoughts.",
    "Motifs recur throughout a work to develop theme and reinforce meaning.",
    "The sonnet form imposes a strict structure that can create productive tension.",
    "Improvisation in jazz allows musicians to create spontaneous melodic variations.",
    "Dramatic irony occurs when the audience knows something the characters do not.",
    "The camera angle in film shapes the viewer's emotional relationship with the subject.",
    "Genre fiction satisfies reader expectations while still allowing for innovation.",
    "Colour symbolism varies across cultures and affects the reading of visual art.",
    "Editing in film controls pacing and can manipulate the perception of time.",

    # ── Health & Medicine ────────────────────────────────────────────────────
    "The immune system distinguishes between self and non-self to fight infections.",
    "Sleep deprivation impairs memory consolidation and executive function.",
    "Vaccines work by training the immune system to recognise pathogen antigens.",
    "Antibiotic resistance is accelerated by the overprescription of antibiotics.",
    "Chronic stress raises cortisol levels, which can damage the hippocampus.",
    "Epigenetic changes alter gene expression without changing the DNA sequence.",
    "Placebos can produce measurable physiological effects due to expectation.",
    "Mental health conditions have both genetic and environmental risk factors.",
    "Nutrition affects not only physical health but also cognitive performance.",
    "Clinical trials use randomised controlled designs to establish causal efficacy.",

    # ── Economy & Business ───────────────────────────────────────────────────
    "Supply and demand curves intersect at the equilibrium price and quantity.",
    "Inflation erodes the purchasing power of money over time.",
    "Compound interest causes wealth to grow exponentially over long periods.",
    "Market failures arise from externalities, public goods, and information asymmetry.",
    "Game theory analyses strategic interactions between rational agents.",
    "Central banks use monetary policy to stabilise inflation and employment.",
    "Network effects make products more valuable as more people use them.",
    "Comparative advantage explains why countries benefit from specialised trade.",
    "Behavioural economics integrates psychological insights into economic models.",
    "Venture capital funds early-stage companies with high growth potential.",

    # ── Environment & Climate ────────────────────────────────────────────────
    "Deforestation reduces biodiversity and accelerates carbon release into the atmosphere.",
    "Ocean acidification threatens coral ecosystems by lowering seawater pH.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power.",
    "The water cycle redistributes freshwater across the Earth through evaporation and precipitation.",
    "Microplastics have been detected in ecosystems from the deep ocean to the Arctic.",
    "Biodiversity loss weakens ecosystem resilience to environmental disturbance.",
    "Carbon capture technologies attempt to remove CO2 directly from the atmosphere.",
    "Permafrost thaw releases methane, a potent greenhouse gas.",
    "Sustainable agriculture seeks to maintain yields while reducing environmental impact.",
    "Urban heat islands raise temperatures in cities relative to surrounding rural areas.",

    # ── Psychology ───────────────────────────────────────────────────────────
    "Classical conditioning associates a neutral stimulus with a meaningful one.",
    "Operant conditioning shapes behaviour through rewards and punishments.",
    "Attachment theory describes how early bonds with caregivers shape later relationships.",
    "The amygdala plays a central role in processing fear and emotional memory.",
    "Self-efficacy beliefs influence how people approach challenges and setbacks.",
    "Social identity theory explains in-group loyalty and out-group discrimination.",
    "Intrinsic motivation drives behaviour through interest and enjoyment rather than reward.",
    "Cognitive dissonance creates discomfort when beliefs and actions conflict.",
    "The availability heuristic judges frequency by how easily examples come to mind.",
    "Personality traits show moderate heritability and remarkable stability over time.",

    # ── Astronomy & Physics ──────────────────────────────────────────────────
    "General relativity describes gravity as the curvature of spacetime by mass and energy.",
    "Dark matter does not interact with light but exerts gravitational effects on galaxies.",
    "The Big Bang model predicts that the universe began in an extremely hot, dense state.",
    "Neutron stars are so dense that a teaspoon of their material would weigh billions of tonnes.",
    "Quantum mechanics is fundamentally probabilistic at the level of individual particles.",
    "The Pauli exclusion principle prevents two fermions from occupying the same quantum state.",
    "Cosmic microwave background radiation is the afterglow of the early universe.",
    "Exoplanets are detected indirectly by the dimming of starlight during transit.",
    "The expansion of the universe is accelerating, driven by dark energy.",
    "Lasers produce coherent light through stimulated emission of photons.",

    # ── Everyday Observations ────────────────────────────────────────────────
    "Learning a new skill requires deliberate practice over an extended period.",
    "Writing clearly demands organising thoughts before committing them to the page.",
    "A good night's sleep resets mood, attention, and problem-solving capacity.",
    "Reading widely broadens vocabulary and exposes readers to diverse perspectives.",
    "Regular physical exercise improves cardiovascular health and mental wellbeing.",
    "Cooking from raw ingredients gives greater control over nutrition and flavour.",
    "Listening attentively in a conversation shows respect and builds trust.",
    "Travelling to unfamiliar places challenges assumptions and fosters empathy.",
    "Keeping a journal externalises thoughts and can reduce rumination.",
    "Boredom can be a productive state that spurs creativity and reflection.",
    "Small habits compounded over time produce large changes in behaviour.",
    "Public speaking improves with practice and familiarity with the material.",
    "Curiosity is a reliable indicator of long-term learning success.",
    "Time management is fundamentally about deciding what not to do.",
    "Collaboration often produces outcomes superior to individual effort.",

    # ── Short conversational / low-complexity ────────────────────────────────
    "How are you doing today?",
    "What is your name?",
    "I would like a coffee, please.",
    "The train arrives at three o'clock.",
    "Could you help me find the library?",
    "It is raining heavily outside.",
    "She smiled and waved from across the street.",
    "He forgot to bring his keys again.",
    "The children played in the garden until sunset.",
    "We need to leave early to avoid traffic.",
    "Please turn off the lights when you leave.",
    "I have not seen that film yet.",
    "The meeting has been rescheduled to Friday.",
    "Can you repeat that more slowly?",
    "It was a pleasure to meet you.",
]


# ---------------------------------------------------------------------------
# CorpusLoader
# ---------------------------------------------------------------------------

class CorpusLoader:
    """Utilities for loading or generating text corpora."""

    @staticmethod
    def diverse_sentences(
        n: int = 1000,
        seed: int = 42,
        min_len: int = 20,
    ) -> List[str]:
        """
        Return up to *n* diverse sentences from the built-in corpus.

        The base corpus has ~750 sentences.  If n > 750, sentences are
        resampled with replacement so you always get exactly n items.

        Parameters
        ----------
        n : int
            Number of sentences to return.
        seed : int
            Random seed for reproducible sampling.
        min_len : int
            Drop sentences shorter than this many characters.
        """
        base = [s for s in _SENTENCES if len(s) >= min_len]
        rng  = random.Random(seed)

        if n <= len(base):
            return rng.sample(base, n)

        # Oversample: shuffle the base list repeatedly
        result: List[str] = []
        while len(result) < n:
            shuffled = base[:]
            rng.shuffle(shuffled)
            result.extend(shuffled)
        return result[:n]

    @staticmethod
    def from_openwebtext(
        n: int = 1000,
        seed: int = 42,
        max_chars_per_sentence: int = 400,
        min_words: int = 6,
    ) -> List[str]:
        """
        Stream *n* sentences from the openwebtext dataset via Hugging Face.

        Requires:
            pip install datasets

        Splits each document into sentences, filters by quality, and
        returns a flat list of n cleaned strings.

        Parameters
        ----------
        n : int
        seed : int
        max_chars_per_sentence : int
            Discard very long sentences to keep token counts manageable.
        min_words : int
            Discard very short fragments.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required for openwebtext loading.\n"
                "Install it with:  pip install datasets"
            ) from exc

        logger.info("Streaming openwebtext (n=%d, seed=%d) …", n, seed)
        ds = load_dataset(
            "openwebtext",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

        sentences: List[str] = []
        for doc in ds:
            raw = doc.get("text", "")
            for sent in _split_sentences(raw):
                sent = sent.strip()
                words = sent.split()
                if len(words) < min_words:
                    continue
                if len(sent) > max_chars_per_sentence:
                    continue
                sentences.append(sent)
                if len(sentences) >= n:
                    break
            if len(sentences) >= n:
                break

        if len(sentences) < n:
            logger.warning(
                "Only collected %d / %d sentences from openwebtext stream.",
                len(sentences), n,
            )
        logger.info("Collected %d sentences from openwebtext.", len(sentences))
        return sentences[:n]

    @staticmethod
    def from_file(
        path: str,
        n: Optional[int] = None,
        seed: int = 42,
        encoding: str = "utf-8",
    ) -> List[str]:
        """
        Load sentences from a plain-text file (one sentence per line).

        Parameters
        ----------
        path : str
        n : int | None
            Maximum sentences to return.  None → return all.
        seed : int
            Random seed for shuffling when n < total.
        encoding : str
        """
        with open(path, encoding=encoding) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if n is None or n >= len(lines):
            return lines
        rng = random.Random(seed)
        return rng.sample(lines, n)

    @staticmethod
    def with_labels(
        sentences: List[str],
        labels: List[str],
    ) -> "tuple[list[str], list[str]]":
        """Validate and return (sentences, labels) as a matched pair."""
        if len(sentences) != len(labels):
            raise ValueError(
                f"len(sentences)={len(sentences)} != len(labels)={len(labels)}"
            )
        return sentences, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter (no external NLP dependency)."""
    parts = _SENTENCE_RE.split(text)
    # Further split on newlines that likely separate sentences
    result = []
    for part in parts:
        for sub in part.split("\n"):
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result
