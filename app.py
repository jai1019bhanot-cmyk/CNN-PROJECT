import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageStat
import pickle
import os

# Page config
st.set_page_config(
    page_title="Cart Detector AI",
    page_icon="🛒",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 700;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
    .empty { background: #1a3a2a; border: 2px solid #22c55e; }
    .filled { background: #3a1a1a; border: 2px solid #ef4444; }
    .not-cart { background: #1a1a3a; border: 2px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    # FIXED: Use __file__ to find model path on Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), "model", "cart_detector_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def extract_features(image, size=(128, 128)):
    """Extract features from uploaded image"""
    img = image.convert('RGB')
    img = img.resize(size)
    gray = img.convert('L')

    # Statistical features
    stat = ImageStat.Stat(img)
    mean_r, mean_g, mean_b = stat.mean[0], stat.mean[1], stat.mean[2]
    std_r, std_g, std_b = stat.stddev[0], stat.stddev[1], stat.stddev[2]

    # Edge detection
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_mean, edge_std = edge_stat.mean[0], edge_stat.stddev[0]

    # Blur detection
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
    diff = np.array(gray) - np.array(blurred)
    blur_score = np.std(diff)

    # Brightness
    brightness = np.array(gray).mean()

    # Histogram features
    hist_r = np.histogram(np.array(img)[:,:,0], bins=8, range=(0, 256))[0]
    hist_g = np.histogram(np.array(img)[:,:,1], bins=8, range=(0, 256))[0]
    hist_b = np.histogram(np.array(img)[:,:,2], bins=8, range=(0, 256))[0]

    features = [mean_r, mean_g, mean_b, std_r, std_g, std_b, 
                edge_mean, edge_std, blur_score, brightness]
    features.extend(hist_r)
    features.extend(hist_g)
    features.extend(hist_b)

    return np.array(features).reshape(1, -1)

# Title
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #f59e0b; font-size: 2.5rem;">🛒 Cart Detector AI</h1>
    <p style="color: #8b949e; font-size: 1.1rem;">
        Upload an image to detect if a cart is <b>Empty</b>, <b>Filled</b>, or <b>Not a Cart</b>
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        model_data = load_model()
        clf = model_data['classifier']
        classes = model_data['classes']

        features = extract_features(image)
        pred = clf.predict(features)[0]
        probs = clf.predict_proba(features)[0]

        confidence = probs[pred] * 100
        class_name = classes[pred]

        # Display results
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)

        if class_name == 'empty_cart':
            st.markdown(f"""
            <div class="result-box empty" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #22c55e;">✅ EMPTY CART</div>
                <div style="font-size: 1.2rem; color: #86efac; margin-top: 0.5rem;">
                    Confidence: {confidence:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
                    This cart needs to be restocked
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif class_name == 'filled_cart':
            st.markdown(f"""
            <div class="result-box filled" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #ef4444;">📦 FILLED CART</div>
                <div style="font-size: 1.2rem; color: #fca5a5; margin-top: 0.5rem;">
                    Confidence: {confidence:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
                    This cart has items in it
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box not-cart" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">❓ NOT A CART</div>
                <div style="font-size: 1.2rem; color: #93c5fd; margin-top: 0.5rem;">
                    Confidence: {confidence:.1f}%
                </div>
                <div style="font-size: 0.9rem; color: #4a5568; margin-top: 0.5rem;">
                    This doesn't appear to be a cart
                </div>
            </div>
            """, unsafe_allow_html=True)

        # All probabilities
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3 style="text-align: center;">All Probabilities</h3>
        </div>
        """, unsafe_allow_html=True)

        for i, cls in enumerate(classes):
            prob = probs[i] * 100
            st.progress(int(prob))
            display_name = cls.replace('_', ' ').title()
            st.markdown(f"<div style='text-align: center;'>{display_name}: {prob:.1f}%</div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Model not found. Please run train_model.py first.")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem;">
        <h3 style="color: #f59e0b;">About</h3>
        <p style="color: #8b949e; font-size: 0.9rem;">
            AI-powered image classifier for warehouse automation.
            Detects whether a cart is empty, filled, or not a cart.
        </p>
        <hr style="border-color: #1c2333;">
        <h4 style="color: #f59e0b;">Classes</h4>
        <ul style="color: #8b949e; font-size: 0.85rem;">
            <li>🟢 Empty Cart</li>
            <li>🔴 Filled Cart</li>
            <li>🔵 Not a Cart</li>
        </ul>
        <hr style="border-color: #1c2333;">
        <p style="color: #4a5568; font-size: 0.75rem;">
            Model: Random Forest<br>
            Features: 34 (color, edges, texture)<br>
            Classes: 3
        </p>
    </div>
    """, unsafe_allow_html=True)
