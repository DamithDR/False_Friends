"""
False Friends Detector — Streamlit UI
======================================
Select a language pair (EN-ES or EN-FR), enter an English sentence and its
translation, then click "Analyze" to see — at the token level — which words are
false friends, highlighted in place on both sides.

Backend integration
-------------------
MODEL_PATHS points at the directories written by run_token_classification.sh
(token_classification.py train --output_dir ...). Override per-pair with the
FF_MODEL_EN_ES / FF_MODEL_EN_FR environment variables. When a model directory
is missing the app falls back to demo mode so the UI still renders.
"""

import os
import html
import streamlit as st

# ── Model paths — match run_token_classification.sh --output_dir ──────────────
MODEL_PATHS = {
    "EN-ES": os.getenv("FF_MODEL_EN_ES", "outputs/token_classification/ff_xlmr_es"),
    "EN-FR": os.getenv("FF_MODEL_EN_FR", "outputs/token_classification/ff_xlmr_fr"),
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
    .ff-sentence {
        font-size: 1.1rem;
        line-height: 2.2rem;
        word-spacing: 0.1rem;
    }
    .ff-highlight {
        background-color: #ff4b4b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .ff-token-ok {
        padding: 2px 3px;
    }
    .result-box {
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 0.95rem;
    }
    .result-positive { background-color: #fff0f0; border-left: 4px solid #ff4b4b; }
    .result-negative { background-color: #f0fff4; border-left: 4px solid #21c55d; }
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


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    """Load (model, tokenizer, max_length) once per path. Cached across reruns."""
    from token_classification import load_model  # noqa: PLC0415
    return load_model(model_path)


# ── Backend: token-level inference, fall back to demo mode ────────────────────
def run_inference(language_pair: str, english: str, other: str):
    """Return (result_dict_or_None, demo_mode, error).

    result_dict has keys source_tokens, target_tokens, source_labels,
    target_labels, source_ff, target_ff (see token_classification.predict_tokens).
    """
    model_path = MODEL_PATHS[language_pair]
    if not os.path.isdir(model_path):
        return None, True, None
    try:
        from token_classification import predict_tokens  # noqa: PLC0415
        model, tokenizer, max_length = get_model(model_path)
        result = predict_tokens(model, tokenizer, english, other, max_length)
        return result, False, None
    except Exception as exc:  # noqa: BLE001
        return None, True, str(exc)


# ── Render a tokenised sentence with per-token highlighting ───────────────────
def render_tokens(tokens: list[str], labels: list[str]) -> str:
    """Highlight tokens whose predicted label is B-FF, by position."""
    spans = []
    for tok, lab in zip(tokens, labels):
        safe = html.escape(tok)
        if lab == "B-FF":
            spans.append(f'<span class="ff-highlight">{safe}</span>')
        else:
            spans.append(f'<span class="ff-token-ok">{safe}</span>')
    return '<span class="ff-sentence">' + " ".join(spans) + "</span>"


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("False Friends Detector")
st.caption(
    "A false friend is a word that looks similar in two languages but means "
    "something different. Enter a sentence pair to detect false friends at the "
    "token level."
)

st.divider()

language_pair = st.radio(
    "Language pair",
    options=["EN-ES", "EN-FR"],
    format_func=lambda x: "English ↔ Spanish" if x == "EN-ES" else "English ↔ French",
    horizontal=True,
)
other_lang_label = "Spanish" if language_pair == "EN-ES" else "French"

st.write("")

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
analyze_btn = st.button("Analyze", type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if analyze_btn:
    english_text = english_text.strip()
    other_text = other_text.strip()

    if not english_text or not other_text:
        st.warning("Please fill in both text boxes before analyzing.")
        st.stop()

    with st.spinner("Analyzing sentence pair…"):
        result, demo_mode, error = run_inference(
            language_pair, english_text, other_text
        )

    st.divider()
    st.subheader("Results")

    if error:
        st.error(f"Inference error: {error}")

    if demo_mode:
        st.markdown(
            '<div class="demo-banner">'
            "⚠️ <strong>Model not loaded</strong> — no trained model was found at "
            f"<code>{MODEL_PATHS[language_pair]}</code>. "
            "Train a model (see <code>run_token_classification.sh</code>) or set the "
            "<code>FF_MODEL_EN_ES</code> / <code>FF_MODEL_EN_FR</code> env var to enable "
            "real inference."
            "</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    source_ff = result["source_ff"]
    target_ff = result["target_ff"]
    has_ff = bool(source_ff or target_ff)

    # ── Sentence-level verdict (derived from token predictions) ───────────────
    if has_ff:
        st.markdown(
            '<div class="result-box result-positive">'
            "🔴 <strong>Verdict:</strong> This sentence pair "
            "<strong>contains false friends</strong>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-box result-negative">'
            "🟢 <strong>Verdict:</strong> No false friends detected in this sentence pair."
            "</div>",
            unsafe_allow_html=True,
        )
    st.write("")

    # ── Token-level highlighted sentences ─────────────────────────────────────
    st.markdown("**Token-level detection**")
    hl_en, hl_other = st.columns(2)
    with hl_en:
        st.caption(f"English — {', '.join(source_ff) if source_ff else 'no false friends'}")
        st.markdown(
            render_tokens(result["source_tokens"], result["source_labels"]),
            unsafe_allow_html=True,
        )
    with hl_other:
        st.caption(f"{other_lang_label} — {', '.join(target_ff) if target_ff else 'no false friends'}")
        st.markdown(
            render_tokens(result["target_tokens"], result["target_labels"]),
            unsafe_allow_html=True,
        )

    st.write("")

    # ── Per-token table (only flagged tokens, with position) ──────────────────
    rows = []
    for i, (tok, lab) in enumerate(zip(result["source_tokens"], result["source_labels"])):
        if lab == "B-FF":
            rows.append({"Side": "English", "Position": i, "Token": tok, "Label": lab})
    for i, (tok, lab) in enumerate(zip(result["target_tokens"], result["target_labels"])):
        if lab == "B-FF":
            rows.append({"Side": other_lang_label, "Position": i, "Token": tok, "Label": lab})

    if rows:
        st.markdown("**Detected false-friend tokens**")
        st.table(rows)

    # ── Full token-by-token breakdown (collapsible) ───────────────────────────
    with st.expander("Show full token-by-token predictions"):
        full_rows = []
        for i, (tok, lab) in enumerate(zip(result["source_tokens"], result["source_labels"])):
            full_rows.append({"Side": "English", "Position": i, "Token": tok, "Label": lab})
        for i, (tok, lab) in enumerate(zip(result["target_tokens"], result["target_labels"])):
            full_rows.append({"Side": other_lang_label, "Position": i, "Token": tok, "Label": lab})
        st.dataframe(full_rows, use_container_width=True, hide_index=True)
