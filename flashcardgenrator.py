"""
flashcard_generator.py
Accepts plain text → chunks it → calls a seq2seq model → returns exam-focused flashcards.

Key changes vs. original:
  - Switched to google/flan-t5-base (instruction-tuned, stronger zero-shot generalisation)
  - Prompts now enforce abstractive, ≤20-word answers in exam-ready form
  - Added abstractive answer summarisation step (compress_answer)
  - Added mnemonic/diagram injection where heuristics detect list-like or causal content
  - Removed purely extractive answer selection

Dependencies:
    pip install transformers torch sentencepiece
"""

import re
import math
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """Naive but robust sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _chunk_text(
    text: str,
    max_sentences_per_chunk: int = 5,
    overlap: int = 1,
) -> List[str]:
    """
    Split text into overlapping chunks of sentences.
    Overlap preserves cross-sentence context at boundaries.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    step = max(1, max_sentences_per_chunk - overlap)
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + max_sentences_per_chunk])
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pipeline():
    """
    Lazy-load FLAN-T5-base (instruction-tuned seq2seq).
    Falls back to a stub pipeline if the model is unavailable.
    Cached at module level so it's only loaded once per process.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        _PIPELINE = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print(f"[flashcard_generator] Loaded model: {model_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[flashcard_generator] WARNING – Could not load HF model ({exc}). "
              "Falling back to extractive stub.")
        _PIPELINE = _STUB_PIPELINE

    return _PIPELINE


_PIPELINE = None  # module-level cache


# ---------------------------------------------------------------------------
# Stub pipeline (testing / offline fallback)
# ---------------------------------------------------------------------------

def _STUB_PIPELINE(inputs, **kwargs):
    """
    Extractive fallback that mimics HF pipeline output format.
    Produces a basic question or answer depending on prompt type.
    """
    results = []
    for inp in inputs if isinstance(inputs, list) else [inputs]:
        prompt = str(inp)
        upper_prompt = prompt.upper()
        context_match = re.search(
            r'\[CONTEXT\]\s*(.*?)(?:\n\[QUESTION\]|\n\[TASK\]|$)',
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        context = context_match.group(1).strip() if context_match else prompt
        sentences = _split_into_sentences(context)
        anchor = max(sentences, key=len) if sentences else context

        if "[QUESTION]" in upper_prompt or "ANSWER THE QUESTION" in upper_prompt:
            # Build a short factual answer from context instead of echoing a question.
            answer_words = re.findall(r'\w+', anchor)
            if len(answer_words) >= 4:
                answer = " ".join(answer_words[:20]).strip() + "."
            else:
                fallback = re.sub(r'\s+', ' ', context).strip()
                answer = (fallback[:120].rstrip(" .") + ".") if fallback else "Key concept from the provided text."
            results.append({"generated_text": answer})
        else:
            clause = anchor.split(",")[0][:110].strip(" .")
            if not clause:
                clause = "the key idea in the text"
            question = f"What is the key concept about {clause}?"
            results.append({"generated_text": question})
    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_QG_PROMPT_TEMPLATE = (
    "[CONTEXT] {chunk}\n"
    "[TASK] Generate one precise exam flashcard question from the context above. "
    "The question must target the single most important concept. "
    "Output only the question, ending with a question mark."
)

_ANSWER_PROMPT_TEMPLATE = (
    "[CONTEXT] {chunk}\n"
    "[QUESTION] {question}\n"
    "[TASK] Answer the question above in 20 words or fewer. "
    "Be direct, use precise terminology, and omit filler words. "
    "Output only the answer."
)

_COMPRESS_PROMPT_TEMPLATE = (
    "[TASK] Rewrite the following sentence as a concise exam fact in 20 words or fewer. "
    "Keep only the core concept, use precise terminology, and remove filler language.\n"
    "[INPUT] {sentence}"
)


def _build_qg_prompt(chunk: str) -> str:
    return _QG_PROMPT_TEMPLATE.format(chunk=chunk)


def _build_answer_prompt(chunk: str, question: str) -> str:
    return _ANSWER_PROMPT_TEMPLATE.format(chunk=chunk, question=question)


def _build_compress_prompt(sentence: str) -> str:
    return _COMPRESS_PROMPT_TEMPLATE.format(sentence=sentence)


def _normalize_instructions(instructions: Optional[str]) -> str:
    if not instructions:
        return ""
    cleaned = instructions.strip()
    if not cleaned:
        return ""
    return f"\n[USER_INSTRUCTIONS] {cleaned}"


# ---------------------------------------------------------------------------
# Abstractive answer summarisation
# ---------------------------------------------------------------------------

def _compress_answer(raw_answer: str, pipeline_fn) -> str:
    """
    If the raw answer exceeds 20 words, use the model to compress it
    into an abstractive ≤20-word exam fact.
    """
    words = re.findall(r'\w+', raw_answer)
    if len(words) <= 20:
        return raw_answer.strip()

    prompt = _build_compress_prompt(raw_answer)
    try:
        out = pipeline_fn(
            [prompt],
            max_new_tokens=48,
            num_beams=4,
            early_stopping=True,
        )
        compressed = (
            out[0][0]["generated_text"]
            if isinstance(out[0], list)
            else out[0]["generated_text"]
        ).strip()
        return compressed if compressed else raw_answer
    except Exception:
        # Hard truncation fallback: keep first 20 words
        return " ".join(words[:20]) + "."


# ---------------------------------------------------------------------------
# Mnemonic / diagram injection
# ---------------------------------------------------------------------------

_CAUSAL_PATTERN = re.compile(
    r'\b(causes?|leads? to|results? in|produces?|converts?|breaks? down|releases?)\b',
    re.IGNORECASE,
)
_LIST_PATTERN = re.compile(r'\b(first|second|third|then|finally|step)\b', re.IGNORECASE)


def _try_add_memory_aid(answer: str, chunk: str) -> Optional[str]:
    """
    Heuristically inject a simple diagram (A → B → C) or mnemonic tag
    when the content is causal / sequential.  Returns None if no aid applies.
    """
    # Causal chain: look for "X causes/leads to Y" in the chunk
    causal_match = _CAUSAL_PATTERN.search(chunk)
    if causal_match:
        # Extract noun phrases around the causal verb as best-effort
        verb_start = causal_match.start()
        left  = chunk[:verb_start].split()[-3:]
        right = chunk[causal_match.end():].split()[:3]
        if left and right:
            a = " ".join(left).strip("., ")
            b = " ".join(right).strip("., ")
            return f"{answer}  [Diagram: {a} → {b}]"

    # Step-wise content: suggest a mnemonic prompt
    if _LIST_PATTERN.search(chunk):
        return f"{answer}  [Tip: create a mnemonic for the ordered steps]"

    return None  # no aid applicable


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(question: str, answer: str, chunk: str) -> float:
    """
    Heuristic confidence score in [0, 1].

    Factors:
    - Non-trivial question length (≥5 words)
    - Ends with a question mark
    - Answer word coverage inside the chunk
    - Answer is not too short (≥4 words preferred)
    - Bonus for abstractive answers (low exact overlap → model paraphrased)
    """
    score = 0.5

    q_words = re.findall(r'\w+', question)
    if len(q_words) >= 5:
        score += 0.1
    if question.strip().endswith("?"):
        score += 0.1

    a_words = set(re.findall(r'\w+', answer.lower()))
    c_words = set(re.findall(r'\w+', chunk.lower()))
    if c_words and a_words:
        coverage = len(a_words & c_words) / len(a_words)
        score += 0.15 * min(coverage, 1.0)
        # Abstractive bonus: if fewer than 60% of answer words are verbatim copies
        if coverage < 0.6:
            score += 0.05

    if len(re.findall(r'\w+', answer)) < 4:
        score -= 0.1

    return round(min(max(score, 0.0), 1.0), 2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_flashcards(
    text: str,
    max_sentences_per_chunk: int = 5,
    overlap: int = 1,
    max_cards: int = 20,
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Main entry point.

    Args:
        text:                    Raw input text (any length).
        max_sentences_per_chunk: Sentences per context window fed to the model.
        overlap:                 Sentence overlap between consecutive chunks.
        max_cards:               Cap on the number of returned flashcards.

    Returns:
        List of dicts with keys:
            question   – exam-focused question string
            answer     – abstractive, ≤20-word answer
            memory_aid – optional diagram / mnemonic string (or None)
            confidence – heuristic score in [0, 1]
    """
    if not text or not text.strip():
        return []

    chunks = _chunk_text(text, max_sentences_per_chunk, overlap)
    if not chunks:
        return []

    pipeline_fn = _load_pipeline()

    # ── Step 1: Generate questions (batched) ────────────────────────────────
    instruction_suffix = _normalize_instructions(instructions)
    qg_prompts = [_build_qg_prompt(chunk) + instruction_suffix for chunk in chunks]

    try:
        qg_outputs = pipeline_fn(
            qg_prompts,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
    except TypeError:
        qg_outputs = pipeline_fn(qg_prompts)

    questions: List[str] = []
    for output in qg_outputs:
        raw = (
            output[0]["generated_text"]
            if isinstance(output, list)
            else output["generated_text"]
        ).strip()
        questions.append(raw)

    # ── Step 2: Generate abstractive answers (batched) ──────────────────────
    answer_prompts = [
        _build_answer_prompt(chunk, q) + instruction_suffix
        for chunk, q in zip(chunks, questions)
    ]

    try:
        ans_outputs = pipeline_fn(
            answer_prompts,
            max_new_tokens=48,
            num_beams=4,
            early_stopping=True,
        )
    except TypeError:
        ans_outputs = pipeline_fn(answer_prompts)

    raw_answers: List[str] = []
    for output in ans_outputs:
        raw = (
            output[0]["generated_text"]
            if isinstance(output, list)
            else output["generated_text"]
        ).strip()
        raw_answers.append(raw)

    # ── Step 3: Compress, deduplicate, and enrich ───────────────────────────
    cards: List[Dict[str, Any]] = []
    seen_questions: set = set()

    for chunk, question, raw_answer in zip(chunks, questions, raw_answers):
        # Deduplicate by normalised question text
        normalised_q = re.sub(r'\W+', ' ', question.lower()).strip()
        if normalised_q in seen_questions:
            continue
        seen_questions.add(normalised_q)

        # Abstractive compression to enforce ≤20-word limit
        answer = _compress_answer(raw_answer, pipeline_fn)

        # Optional memory aid (diagram or mnemonic hint)
        memory_aid = _try_add_memory_aid(answer, chunk)

        confidence = _compute_confidence(question, answer, chunk)

        cards.append(
            {
                "question":   question,
                "answer":     answer,
                "memory_aid": memory_aid,
                "confidence": confidence,
            }
        )

        if len(cards) >= max_cards:
            break

    return cards


# ---------------------------------------------------------------------------
# Quick smoke-test  (python flashcard_generator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = """
    Photosynthesis is the process by which green plants convert sunlight into food.
    It occurs in the chloroplasts, using chlorophyll to absorb light energy.
    The overall reaction combines carbon dioxide and water to produce glucose and oxygen.
    This process is fundamental to almost all life on Earth, forming the base of most food chains.
    Cellular respiration is the reverse process, where glucose is broken down to release energy.
    Mitochondria are the organelles responsible for cellular respiration in eukaryotic cells.
    ATP, or adenosine triphosphate, is the primary energy currency of the cell.
    """

    import json
    cards = generate_flashcards(sample, max_cards=5)
    print(json.dumps(cards, indent=2))
