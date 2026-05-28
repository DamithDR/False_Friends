"""
False Friends Detector — Streamlit UI
======================================
Select a language pair (EN-ES or EN-FR), enter an English sentence and its
translation, then click "Analyze" to see whether the pair contains false
friends and, if so, which words they are.

Backend integration
-------------------
Set the MODEL_PATHS dict below to wherever your trained models live.
When a model directory is present the app calls token_classification.predict();
otherwise it enters demo-mode so the UI can still be developed and tested.
"""

import os
import re
import streamlit as st

# ── Model paths — update once models are trained ──────────────────────────────
MODEL_PATHS = {
    "EN-ES": os.getenv("FF_MODEL_EN_ES", "./ff_model_en_es"),
    "EN-FR": os.getenv("FF_MODEL_EN_FR", "./ff_model_en_fr"),
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="False Friends Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .ff-highlight {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .ff-sentence {
        font-size: 1.1rem;
        line-height: 2rem;
        word-spacing: 0.15rem;
    }
    .result-box {
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 0.95rem;
    }
    .result-positive {
        background-color: #fff0f0;
        border-left: 4px solid #ff4b4b;
    }
    .result-negative {
        background-color: #f0fff4;
        border-left: 4px solid #21c55d;
    }
    .demo-banner {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 16px;
        margin-bottom: 12px;
        font-size: 0.85rem;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: highlight false-friend tokens in a sentence ──────────────────────
def highlight_tokens(sentence: str, false_friends: list[str]) -> str:
    """Return HTML with false-friend words wrapped in a highlight span."""
    if not false_friends:
        return f'<span class="ff-sentence">{sentence}</span>'

    # Build a regex that matches any of the false-friend words (case-insensitive,
    # whole-word boundary) and wraps them in a highlight span.
    escaped = [re.escape(w) for w in false_friends]
    pattern = r"\b(" + "|".join(escaped) + r")\b"

    def replacer(m):
        return f'<span class="ff-highlight">{m.group(0)}</span>'

    highlighted = re.sub(pattern, replacer, sentence, flags=re.IGNORECASE)
    return f'<span class="ff-sentence">{highlighted}</span>'


# ── Backend: attempt real inference, fall back to demo mode ──────────────────
def run_inference(language_pair: str, english: str, other: str):
    """
    Returns (source_ff, target_ff, demo_mode).

    When the trained model is present the real predict() function is called.
    Otherwise demo_mode=True is returned with empty lists so the UI can show
    a 'model not loaded' notice instead of crashing.
    """
    model_path = MODEL_PATHS[language_pair]

    if not os.path.isdir(model_path):
        # Model not trained / path not found — demo mode
        return [], [], True

    # ── Real inference ────────────────────────────────────────────────────────
    # TODO: swap the import below for your sentence-level classifier once it is
    #       ready.  The token_classification.predict() already returns both the
    #       token-level false-friend words AND implicitly tells you whether the
    #       sentence pair contains false friends (non-empty lists → has FF).
    try:
        from token_classification import predict  # noqa: PLC0415

        result = predict(model_path=model_path, source=english, target=other)
        return result["source_ff"], result["target_ff"], False
    except Exception as exc:  # noqa: BLE001
        st.error(f"Inference error: {exc}")
        return [], [], True


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("False Friends Detector")
st.caption(
    "A false friend is a word that looks similar in two languages but means "
    "something different. Enter a sentence pair to check for false friends."
)

st.divider()

# Language pair selector
language_pair = st.radio(
    "Language pair",
    options=["EN-ES", "EN-FR"],
    format_func=lambda x: "English ↔ Spanish" if x == "EN-ES" else "English ↔ French",
    horizontal=True,
)

other_lang_label = "Spanish" if language_pair == "EN-ES" else "French"

st.write("")  # spacing

# Text input columns
col_en, col_other = st.columns(2)

with col_en:
    st.markdown("**English sentence**")
    english_text = st.text_area(
        label="English sentence",
        placeholder="e.g. This is a sensible solution.",
        height=140,
        label_visibility="collapsed",
        key="english_input",
    )

with col_other:
    st.markdown(f"**{other_lang_label} sentence**")
    other_text = st.text_area(
        label=f"{other_lang_label} sentence",
        placeholder=(
            "e.g. Esta es una solución sensible."
            if language_pair == "EN-ES"
            else "e.g. C'est une solution sensible."
        ),
        height=140,
        label_visibility="collapsed",
        key="other_input",
    )

st.write("")

analyze_btn = st.button("Analyze", type="primary", use_container_width=False)

# ── Results ───────────────────────────────────────────────────────────────────
if analyze_btn:
    english_text = english_text.strip()
    other_text = other_text.strip()

    if not english_text or not other_text:
        st.warning("Please fill in both text boxes before analyzing.")
    else:
        with st.spinner("Analyzing sentence pair…"):
            source_ff, target_ff, demo_mode = run_inference(
                language_pair, english_text, other_text
            )

        st.divider()
        st.subheader("Results")

        # Demo-mode banner
        if demo_mode:
            st.markdown(
                '<div class="demo-banner">'
                "⚠️ <strong>Model not loaded</strong> — no trained model was found at "
                f"<code>{MODEL_PATHS[language_pair]}</code>. "
                "The results below are placeholders. Train or point to your model to "
                "enable real inference."
                "</div>",
                unsafe_allow_html=True,
            )

        has_ff = bool(source_ff or target_ff)

        # ── Sentence-level verdict ────────────────────────────────────────────
        if demo_mode:
            verdict_html = (
                '<div class="result-box result-positive">'
                "🔴 <strong>Verdict:</strong> (demo) False friends may be present — "
                "load a trained model for a real result."
                "</div>"
            )
        elif has_ff:
            verdict_html = (
                '<div class="result-box result-positive">'
                "🔴 <strong>Verdict:</strong> This sentence pair <strong>contains false friends</strong>."
                "</div>"
            )
        else:
            verdict_html = (
                '<div class="result-box result-negative">'
                "🟢 <strong>Verdict:</strong> No false friends detected in this sentence pair."
                "</div>"
            )

        st.markdown(verdict_html, unsafe_allow_html=True)
        st.write("")

        # ── Token-level highlights ────────────────────────────────────────────
        if has_ff or demo_mode:
            # In demo mode show placeholder words so the highlight styling is visible
            demo_source_ff = ["sensible"] if demo_mode else []
            demo_target_ff = (
                ["sensible"] if (demo_mode and language_pair == "EN-ES") else
                ["sensible"] if demo_mode else []
            )
            display_source_ff = source_ff if not demo_mode else demo_source_ff
            display_target_ff = target_ff if not demo_mode else demo_target_ff

            # Use demo placeholder sentences when in demo mode so the highlight
            # renders on something even if the user left the boxes empty.
            display_english = english_text or "This is a sensible solution."
            display_other = other_text or (
                "Esta es una solución sensible."
                if language_pair == "EN-ES"
                else "C'est une solution sensible."
            )

            st.markdown("**Highlighted false friends**")

            hl_col_en, hl_col_other = st.columns(2)

            with hl_col_en:
                st.markdown(
                    f"*English ({', '.join(display_source_ff) or 'none'})*",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_tokens(display_english, display_source_ff),
                    unsafe_allow_html=True,
                )

            with hl_col_other:
                st.markdown(
                    f"*{other_lang_label} ({', '.join(display_target_ff) or 'none'})*",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    highlight_tokens(display_other, display_target_ff),
                    unsafe_allow_html=True,
                )

            st.write("")

            # Word-level summary table
            all_ff_words = []
            for w in display_source_ff:
                all_ff_words.append({"Side": "English", "False friend word": w})
            for w in display_target_ff:
                all_ff_words.append({"Side": other_lang_label, "False friend word": w})

            if all_ff_words:
                st.markdown("**Detected false friend words**")
                st.table(all_ff_words)
