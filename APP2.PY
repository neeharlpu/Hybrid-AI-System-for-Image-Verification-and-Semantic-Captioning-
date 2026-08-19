import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

from modules.preprocessing import get_transform
from modules.classifier import load_model, predict
from modules.attention_map import generate_gradcam
from modules.blip_model import load_blip, generate_caption
from modules.multilingual import load_translation_model, translate, LANG_CODES

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hybrid AI — Image Verification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# GLOBAL CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #050608;
    --surface:   #0c0e14;
    --surface2:  #111420;
    --border:    #1c2135;
    --border2:   #252d45;
    --accent:    #6C63FF;
    --accent2:   #a78bfa;
    --teal:      #2dd4bf;
    --red:       #f43f5e;
    --green:     #10b981;
    --muted:     #4a5578;
    --sub:       #8892b0;
    --text:      #ccd6f6;
    --bright:    #e6f1ff;
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }

/* ── GRID BACKGROUND ── */
[data-testid="stAppViewContainer"] > .main {
    background:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    background-color: var(--bg);
}

[data-testid="stAppViewContainer"] > .main::before {
    content: '';
    position: fixed;
    inset: 0;
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(108,99,255,0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.25rem;
}

.sb-logo-wrap { margin-bottom: 2rem; }

.sb-logo-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), var(--teal));
    font-size: 20px;
    margin-bottom: 12px;
}

.sb-logo-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--bright);
    text-transform: uppercase;
    line-height: 1.3;
}

.sb-logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: 0.06em;
}

.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 6px 0 8px 0;
    border-top: 1px solid var(--border);
    margin-top: 16px;
}


[data-testid="stSidebar"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    color: var(--sub) !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ── HERO ── */
.hero {
    padding: 52px 0 44px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 48px;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 100px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 20px;
}

.hero-tag-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 48px;
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: var(--bright);
    margin: 0;
}

.hero-title .hl {
    background: linear-gradient(90deg, var(--accent), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-title-line2 {
    font-family: 'Outfit', sans-serif;
    font-size: 48px;
    font-weight: 300;
    color: var(--sub);
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 0 0 20px 0;
}

.hero-desc {
    font-size: 14px;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.8;
}

/* ── DIVIDER LABELS ── */
.divlabel {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.divlabel-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
}
.divlabel-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 14px !important;
    padding: 28px 24px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── IMAGE FRAME ── */
.img-frame {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px;
}
[data-testid="stImage"] img { border-radius: 10px !important; }

/* ── VERDICT CARD ── */
.verdict {
    position: relative;
    border-radius: 16px;
    border: 1px solid var(--border);
    padding: 28px 28px 24px;
    margin-bottom: 14px;
    overflow: hidden;
    background: var(--surface);
}
.verdict::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.verdict.real::before { background: linear-gradient(90deg, var(--green), #34d399, var(--teal)); }
.verdict.fake::before { background: linear-gradient(90deg, var(--red), #fb7185); }

.verdict-glow-real {
    position: absolute; top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 80px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.14), transparent 70%);
    pointer-events: none;
}
.verdict-glow-fake {
    position: absolute; top: -40px; left: 50%;
    transform: translateX(-50%);
    width: 200px; height: 80px;
    background: radial-gradient(ellipse, rgba(244,63,94,0.14), transparent 70%);
    pointer-events: none;
}
.verdict-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.verdict-value {
    font-family: 'Outfit', sans-serif;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
}
.verdict-value.real { color: var(--green); }
.verdict-value.fake { color: var(--red); }

.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
}
.conf-bar-bg {
    flex: 1;
    height: 3px;
    background: var(--border2);
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 3px;
    border-radius: 100px;
}
.conf-bar-fill.real { background: linear-gradient(90deg, var(--green), var(--teal)); }
.conf-bar-fill.fake { background: linear-gradient(90deg, var(--red), #fb7185); }
.conf-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--sub);
    white-space: nowrap;
}

/* ── INFO CARD ── */
.ic {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.ic::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--teal));
    border-radius: 3px 0 0 3px;
}
.ic-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ic-label::before { content: '//'; color: var(--muted); }
.ic-body {
    font-size: 14px;
    color: var(--text);
    line-height: 1.75;
}

/* ── XAI PANELS ── */
.xai-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
}
.xai-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
}
.xai-panel-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
}
.xai-panel-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 6px var(--teal);
}
.xai-panel-body { padding: 10px; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
    border: 1px dashed var(--border2);
    border-radius: 20px;
    background: var(--surface);
    text-align: center;
    margin-top: 8px;
}

.empty-icon-wrap { position: relative; margin-bottom: 24px; }

.empty-icon-ring {
    width: 80px; height: 80px;
    border-radius: 50%;
    border: 1px solid var(--border2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    background: var(--surface2);
    position: relative;
    z-index: 1;
}
.empty-icon-pulse {
    position: absolute;
    inset: -8px;
    border-radius: 50%;
    border: 1px solid rgba(108,99,255,0.25);
    animation: ring-pulse 2.4s ease-out infinite;
}
@keyframes ring-pulse {
    0%   { transform: scale(0.9); opacity: 0.8; }
    100% { transform: scale(1.4); opacity: 0; }
}
.empty-title {
    font-family: 'Outfit', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--bright);
    margin-bottom: 8px;
}
.empty-body {
    font-size: 13px;
    color: var(--muted);
    max-width: 300px;
    line-height: 1.7;
}
.empty-steps {
    display: flex;
    gap: 12px;
    margin-top: 28px;
    flex-wrap: wrap;
    justify-content: center;
}
.empty-step {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--sub);
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 6px;
}
.step-num {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.3);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    color: var(--accent2);
    flex-shrink: 0;
}

/* ── MISC ── */
.stSpinner > div {
    border-color: var(--accent) transparent transparent transparent !important;
}
hr { border-color: var(--border) !important; }
.block-container { padding-top: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# DEVICE
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_core():
    classifier = load_model(device)
    processor, blip_model = load_blip()
    return classifier, processor, blip_model

@st.cache_resource
def load_translator():
    tokenizer, model = load_translation_model()
    model.to("cpu")
    return tokenizer, model

classifier, processor, blip_model = load_core()
tokenizer, trans_model = load_translator()
transform = get_transform()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo-wrap">
        <div class="sb-logo-icon">🔬</div>
        <div class="sb-logo-title">Hybrid AI System</div>
        <div class="sb-logo-sub">Image Verification v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Configuration</div>', unsafe_allow_html=True)

    lang = st.selectbox("Output Language", list(LANG_CODES.keys()))
    show_attention = st.checkbox("Show Attention Maps", value=True)



# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-tag">
        <div class="hero-tag-dot"></div>
        Deep Learning · Computer Vision · NLP
    </div>
    <div class="hero-title"><span class="hl">Hybrid AI System</span></div>
    <div class="hero-title-line2">for Image Verification &amp; Semantic Captioning</div>
    <div class="hero-desc">
        A multi-model pipeline that authenticates image integrity,
        generates natural language descriptions, and surfaces
        explainable visual evidence — all in a single inference pass.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# UPLOAD
# =========================
st.markdown("""
<div class="divlabel">
    <span class="divlabel-text">01 · Upload</span>
    <div class="divlabel-line"></div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "DROP AN IMAGE FILE HERE  ·  JPG · PNG · WEBP",
    type=["jpg", "png", "jpeg", "webp"],
    label_visibility="visible"
)

# =========================
# MAIN CONTENT
# =========================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="divlabel">
        <span class="divlabel-text">02 · Analysis</span>
        <div class="divlabel-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.35, 1], gap="large")

    with col1:
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        tensor = transform(image).unsqueeze(0).to(device)
        label, conf = predict(tensor, classifier)
        cls = "real" if label == "REAL" else "fake"
        icon = "✦" if label == "REAL" else "✗"

        st.markdown(f"""
        <div class="verdict {cls}">
            <div class="verdict-glow-{cls}"></div>
            <div class="verdict-eyebrow">Authenticity Verdict</div>
            <div class="verdict-value {cls}">{icon} {label}</div>
            <div class="conf-row">
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill {cls}" style="width:{conf*100:.1f}%"></div>
                </div>
                <div class="conf-pct">{conf*100:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Running BLIP captioning..."):
            caption = generate_caption(image, processor, blip_model)

        st.markdown(f"""
        <div class="ic">
            <div class="ic-label">Semantic Caption</div>
            <div class="ic-body">{caption}</div>
        </div>
        """, unsafe_allow_html=True)

        if lang != "English":
            with st.spinner(f"Translating → {lang}..."):
                translated = translate(caption, tokenizer, trans_model, LANG_CODES[lang])
            st.markdown(f"""
            <div class="ic">
                <div class="ic-label">Translation · {lang}</div>
                <div class="ic-body">{translated}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── EXPLAINABILITY ──
    if show_attention:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="divlabel">
            <span class="divlabel-text">03 · Model Explainability (GradCAM)</span>
            <div class="divlabel-line"></div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Computing GradCAM attention maps..."):
            overlay, cam, highlight = generate_gradcam(image, classifier, transform, device)

        cam_color = cv2.applyColorMap(
            (cam * 255).astype("uint8"), cv2.COLORMAP_INFERNO
        )
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

        c1, c2, c3 = st.columns(3, gap="medium")
        panels = [
            (c1, "GradCAM Overlay", overlay),
            (c2, "Key Regions",     highlight.astype("uint8")),
            (c3, "Thermal Heatmap", cam_color),
        ]

        for col, title, img_data in panels:
            with col:
                st.markdown(f"""
                <div class="xai-panel">
                    <div class="xai-panel-header">
                        <div class="xai-panel-title">{title}</div>
                        <div class="xai-panel-dot"></div>
                    </div>
                    <div class="xai-panel-body">
                """, unsafe_allow_html=True)
                st.image(img_data, use_container_width=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon-wrap">
            <div class="empty-icon-ring">🔬</div>
            <div class="empty-icon-pulse"></div>
        </div>
        <div class="empty-title">Ready for Analysis</div>
        <div class="empty-body">
            Upload an image to run the full hybrid AI pipeline —
            verification, captioning, and explainability in one pass.
        </div>
        <div class="empty-steps">
            <div class="empty-step"><span class="step-num">1</span> Upload Image</div>
            <div class="empty-step"><span class="step-num">2</span> Run Inference</div>
            <div class="empty-step"><span class="step-num">3</span> Inspect Results</div>
        </div>
    </div>
    """, unsafe_allow_html=True)