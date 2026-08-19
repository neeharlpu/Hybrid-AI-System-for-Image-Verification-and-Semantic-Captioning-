import streamlit as st
import torch
from PIL import Image
import cv2

from modules.preprocessing import get_transform
from modules.classifier import load_model, predict
from modules.attention_map import generate_gradcam
from modules.blip_model import load_blip, generate_caption
from modules.multilingual import load_translation_model, translate, LANG_CODES

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# CSS (MINIMAL + SAFE)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #ffffff;
}

/* Header */
.header-title {
    font-size: 36px;
    font-weight: 600;
}

.header-sub {
    color: #9aa0a6;
    margin-bottom: 20px;
}

/* Prediction colors */
.real {
    color: #00c853;
    font-size: 28px;
    font-weight: 600;
}

.fake {
    color: #ff3d00;
    font-size: 28px;
    font-weight: 600;
}
</style>

<div class="header-title">
Hybrid AI System for Image Verification and Semantic Captioning
</div>
<div class="header-sub">
Image authenticity detection, captioning, multilingual translation, and explainable AI
</div>
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
st.sidebar.title("Controls")
lang = st.sidebar.selectbox("Language", list(LANG_CODES.keys()))
show_attention = st.sidebar.checkbox("Show Attention Maps", value=True)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1.3, 1])

    # IMAGE
    with col1:
        st.image(image, use_container_width=True)

    # RESULTS
    with col2:
        tensor = transform(image).unsqueeze(0).to(device)
        label, conf = predict(tensor, classifier)

        st.subheader("Prediction")

        if label == "REAL":
            st.markdown(f"<div class='real'>{label}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='fake'>{label}</div>", unsafe_allow_html=True)

        st.progress(conf)
        st.caption(f"{conf*100:.2f}% confidence")

        # Caption
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, processor, blip_model)

        st.subheader("Caption")
        st.write(caption)

        # Translation
        st.subheader("Translation")

        if lang != "English":
            translated = translate(caption, tokenizer, trans_model, LANG_CODES[lang])
        else:
            translated = caption

        st.write(translated)

    # =========================
    # ATTENTION MAP
    # =========================
    if show_attention:
        st.subheader("Model Explainability")

        overlay, cam, highlight = generate_gradcam(
            image, classifier, transform, device
        )

        cam_color = cv2.applyColorMap(
            (cam * 255).astype("uint8"),
            cv2.COLORMAP_INFERNO
        )
        cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption("Overlay")
            st.image(overlay)

        with c2:
            st.caption("Important Regions")
            st.image(highlight.astype("uint8"))

        with c3:
            st.caption("Heatmap")
            st.image(cam_color)