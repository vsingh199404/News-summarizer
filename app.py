"""
News Article Text Summarization - Streamlit Web Application
BITS OSHA Virtual Lab Assignment

A premium Streamlit application for news article summarization
using an Encoder-Decoder Transformer (T5) model.
"""

import streamlit as st
import os
import sys
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add current directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import NewsSummarizer
from preprocess import clean_text, load_sample_articles, extract_text_from_file
from evaluate import compute_rouge, compute_rouge_batch

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Summarizer AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.92);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd5e1; }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    .gradient-text {
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .gradient-text-secondary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.35);
    }

    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #00d2ff !important;
        box-shadow: 0 0 0 3px rgba(0, 210, 255, 0.15) !important;
    }

    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #0083B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(102, 126, 234, 0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(102, 126, 234, 0.7); }

    .summary-box {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.08), rgba(102, 126, 234, 0.08));
        padding: 24px;
        border-radius: 14px;
        border-left: 4px solid #00d2ff;
        font-size: 1.05em;
        line-height: 1.8;
        color: #e2e8f0;
        margin: 10px 0;
    }
    .ref-box {
        background: rgba(102, 126, 234, 0.06);
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        font-size: 0.95em;
        line-height: 1.7;
        color: #cbd5e1;
        margin: 10px 0;
    }

    .team-card {
        background: rgba(255, 255, 255, 0.04);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .team-member {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .team-member:last-child { border-bottom: none; }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-blue { background: rgba(0, 210, 255, 0.15); color: #00d2ff; }
    .badge-purple { background: rgba(102, 126, 234, 0.15); color: #667eea; }

    .rouge-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    .rouge-label { color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .rouge-value { font-size: 1.6rem; font-weight: 800; }
    .rouge-value.blue { background: linear-gradient(135deg, #00d2ff, #0083B0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .rouge-value.purple { background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .rouge-value.cyan { background: linear-gradient(135deg, #3a7bd5, #00d2ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .rouge-detail { color: #64748b; font-size: 0.72rem; margin-top: 4px; }

    hr { border: none; border-top: 1px solid rgba(255, 255, 255, 0.06); margin: 16px 0; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    fine_tuned_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model')
    try:
        if os.path.exists(fine_tuned_path) and os.path.exists(os.path.join(fine_tuned_path, 'config.json')):
            return NewsSummarizer(fine_tuned_path=fine_tuned_path)
        else:
            return NewsSummarizer(model_name='t5-small')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def get_sample_articles():
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_articles')
    return load_sample_articles(sample_dir)


# ─── Main Application ───────────────────────────────────────────────────────
def main():
    apply_custom_css()

    # Session state
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'rouge_scores' not in st.session_state:
        st.session_state.rouge_scores = None
    if 'gen_time' not in st.session_state:
        st.session_state.gen_time = 0
    if 'orig_words' not in st.session_state:
        st.session_state.orig_words = 0
    if 'sum_words' not in st.session_state:
        st.session_state.sum_words = 0

    model = load_model()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h1 class='gradient-text'>📰 News Summarizer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; margin-top:-10px; font-size:0.9em;'>Encoder-Decoder Transformer (T5)</p>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 👨‍💻 Team")
        st.markdown("""
        <div class='team-card'>
            <div class='team-member'>
                <div><b>Vishal Singh Tomar</b><br><span style='color:#94a3b8;font-size:0.8em;'>2025AA05331</span></div>
                <span class='badge badge-blue'>40%</span>
            </div>
            <div class='team-member'>
                <div><b>Vivek Sharma</b><br><span style='color:#94a3b8;font-size:0.8em;'>2025AA05588</span></div>
                <span class='badge badge-purple'>30%</span>
            </div>
            <div class='team-member'>
                <div><b>Yash</b><br><span style='color:#94a3b8;font-size:0.8em;'>2024AD05399</span></div>
                <span class='badge badge-purple'>30%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        with st.expander("Model Architecture"):
            if model:
                info = model.get_model_info()
                for k, v in info.items():
                    label = k.replace('_', ' ').title()
                    st.markdown(f"**{label}:** {v:,}" if isinstance(v, int) and v > 1000 else f"**{label}:** {v}")

        st.markdown(
            "<p style='font-size:0.75em; color:#475569; text-align:center; margin-top:20px;'>"
            "BITS OSHA Virtual Lab<br>News Article Summarization</p>",
            unsafe_allow_html=True
        )

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h2 class='gradient-text' style='text-align:center; margin-bottom:5px;'>"
        "News Article Summarization Engine</h2>"
        "<p style='text-align:center; color:#94a3b8; margin-bottom:30px;'>"
        "Paste or upload an article, get a concise summary with ROUGE evaluation</p>",
        unsafe_allow_html=True
    )

    # ═══════════════════════════════════════════════════════════════════════
    #   SECTION 1:  INPUT
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='gradient-text-secondary'>📥 Input Article</h3>", unsafe_allow_html=True)

    input_method = st.radio(
        "input_method",
        ["Paste Text", "Upload File", "Sample Article"],
        horizontal=True, label_visibility="collapsed"
    )

    text_to_summarize = ""
    reference_summary = ""

    if input_method == "Paste Text":
        text_to_summarize = st.text_area(
            "Paste your news article:",
            height=250,
            placeholder="Paste a news article here to summarize...",
            key="text_input"
        )

    elif input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            try:
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_temp')
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                text_to_summarize = extract_text_from_file(temp_path)
                os.remove(temp_path)
                st.success(f"Loaded **{uploaded_file.name}** ({len(text_to_summarize.split())} words)")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif input_method == "Sample Article":
        samples = get_sample_articles()
        if samples:
            sample_titles = [s['title'] for s in samples]
            selected = st.selectbox("Choose an article:", sample_titles)
            idx = sample_titles.index(selected)
            text_to_summarize = samples[idx]['content']
            reference_summary = samples[idx].get('reference', '')

    # Show input preview for file/sample
    if text_to_summarize and input_method != "Paste Text":
        with st.expander("Preview input article", expanded=False):
            st.write(text_to_summarize)

    generate_btn = st.button("Generate Summary", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    #   SECTION 2:  RESULTS  (summary + metrics + ROUGE)
    # ═══════════════════════════════════════════════════════════════════════
    if generate_btn and text_to_summarize:
        if len(text_to_summarize.strip()) < 50:
            st.warning("Text too short — please provide at least 50 characters.")
        elif not model:
            st.error("Model failed to load.")
        else:
            with st.spinner("Generating summary..."):
                t0 = time.time()
                cleaned = clean_text(text_to_summarize)
                summary = model.summarize(
                    cleaned,
                    max_output_length=150,
                    min_output_length=30,
                    num_beams=4
                )
                gen_time = time.time() - t0

                st.session_state.summary = summary
                st.session_state.gen_time = gen_time
                st.session_state.orig_words = len(text_to_summarize.split())
                st.session_state.sum_words = len(summary.split())

                # Compute ROUGE if reference provided
                if reference_summary.strip():
                    st.session_state.rouge_scores = compute_rouge(summary, reference_summary.strip())
                else:
                    st.session_state.rouge_scores = None

    # Display results if we have a summary
    if st.session_state.summary:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        # ── Generated Summary ────────────────────────────────────────────
        st.markdown("<h3 class='gradient-text-secondary'>📤 Generated Summary</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='summary-box'>{st.session_state.summary}</div>", unsafe_allow_html=True)

        # ── Quick Metrics Row ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        orig = st.session_state.orig_words
        summ = st.session_state.sum_words
        compression = (1 - summ / max(orig, 1)) * 100
        m1.metric("Original", f"{orig} words")
        m2.metric("Summary", f"{summ} words")
        m3.metric("Compressed", f"{compression:.0f}%")
        m4.metric("Time", f"{st.session_state.gen_time:.2f}s")

        # ── ROUGE Evaluation ─────────────────────────────────────────────
        scores = st.session_state.rouge_scores
        if scores:
            st.markdown("---", unsafe_allow_html=True)
            st.markdown("<h3 class='gradient-text-secondary'>📊 ROUGE Evaluation</h3>", unsafe_allow_html=True)

            r1 = scores['rouge1']['f1'] * 100
            r2 = scores['rouge2']['f1'] * 100
            rl = scores['rougeL']['f1'] * 100

            # Score cards
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class='rouge-card'>
                <div class='rouge-label'>ROUGE-1 (Unigram)</div>
                <div class='rouge-value blue'>{r1:.2f}%</div>
                <div class='rouge-detail'>P: {scores['rouge1']['precision']:.3f} &nbsp; R: {scores['rouge1']['recall']:.3f}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class='rouge-card'>
                <div class='rouge-label'>ROUGE-2 (Bigram)</div>
                <div class='rouge-value purple'>{r2:.2f}%</div>
                <div class='rouge-detail'>P: {scores['rouge2']['precision']:.3f} &nbsp; R: {scores['rouge2']['recall']:.3f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class='rouge-card'>
                <div class='rouge-label'>ROUGE-L (LCS)</div>
                <div class='rouge-value cyan'>{rl:.2f}%</div>
                <div class='rouge-detail'>P: {scores['rougeL']['precision']:.3f} &nbsp; R: {scores['rougeL']['recall']:.3f}</div>
            </div>""", unsafe_allow_html=True)

            # Bar chart
            with st.expander("Score visualization", expanded=False):
                fig, ax = plt.subplots(figsize=(7, 3.5))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')

                labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
                vals = [r1, r2, rl]
                colors = ['#00d2ff', '#667eea', '#3a7bd5']

                bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor='white', linewidth=0.5, alpha=0.9)
                ax.set_ylabel('F1 (%)', color='white', fontsize=11, fontweight='bold')
                ax.set_ylim(0, max(max(vals) * 1.35, 10))
                ax.tick_params(colors='white', labelsize=10)
                ax.spines['bottom'].set_color('rgba(255,255,255,0.3)')
                ax.spines['left'].set_color('rgba(255,255,255,0.3)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.yaxis.grid(True, alpha=0.08, color='white')

                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 6), textcoords="offset points",
                                ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.markdown(
                "<p style='color:#64748b; font-size:0.85em; margin-top:10px;'>"
                "ROUGE evaluation requires a human-written reference summary for comparison. "
                "Select a <b>Sample Article</b> to see ROUGE scores, or use the <b>Batch Demo</b> below.</p>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    #   SECTION 3:  BATCH DEMO
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='gradient-text-secondary'>🔬 Batch Demo — All Sample Articles</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.9em;'>Summarize all 5 sample articles at once and view aggregate ROUGE metrics.</p>", unsafe_allow_html=True)

    if st.button("Run Batch Demo", use_container_width=True):
        samples = get_sample_articles()
        if not samples:
            st.warning("No sample articles found.")
        elif not model:
            st.error("Model not loaded.")
        else:
            progress = st.progress(0, text="Starting...")
            results = []
            all_preds, all_refs = [], []

            for i, sample in enumerate(samples):
                progress.progress((i + 1) / len(samples), text=f"Article {i+1}/{len(samples)}: {sample['title'][:40]}...")
                cleaned = clean_text(sample['content'])
                t0 = time.time()
                summary = model.summarize(cleaned, max_output_length=150, min_output_length=30, num_beams=4)
                elapsed = time.time() - t0
                ref = sample.get('reference', '')

                row = {
                    "Title": sample['title'],
                    "Words": f"{len(sample['content'].split())} -> {len(summary.split())}",
                    "Time": f"{elapsed:.1f}s",
                }

                if ref:
                    sc = compute_rouge(summary, ref)
                    if sc:
                        row["R1"] = f"{sc['rouge1']['f1']*100:.1f}"
                        row["R2"] = f"{sc['rouge2']['f1']*100:.1f}"
                        row["RL"] = f"{sc['rougeL']['f1']*100:.1f}"
                        all_preds.append(summary)
                        all_refs.append(ref)

                results.append(row)

                with st.expander(f"{sample['title']}"):
                    lc, rc = st.columns(2)
                    with lc:
                        st.markdown("**Generated:**")
                        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
                    with rc:
                        st.markdown("**Reference:**")
                        st.markdown(f"<div class='ref-box'>{ref if ref else 'N/A'}</div>", unsafe_allow_html=True)

            progress.progress(1.0, text="Done!")

            # Results table
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

            # Average ROUGE
            if all_preds:
                avg = compute_rouge_batch(all_preds, all_refs)
                if avg:
                    st.markdown("#### Average ROUGE Scores")
                    ac1, ac2, ac3 = st.columns(3)
                    ac1.metric("ROUGE-1 F1", f"{avg['rouge1']['f1']*100:.2f}%")
                    ac2.metric("ROUGE-2 F1", f"{avg['rouge2']['f1']*100:.2f}%")
                    ac3.metric("ROUGE-L F1", f"{avg['rougeL']['f1']*100:.2f}%")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
