import streamlit as st
import cv2
import numpy as np
import pickle

from tensorflow.keras.models import load_model

from label_map import label_map

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Ancient Tamil OCR",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0f172a;
        color: white;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b
        );
    }

    h1 {
        color: #f8fafc;
        text-align: center;
        font-size: 3rem !important;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }

    .prediction-box {
        background: linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );
        padding: 20px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-top: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.35);
    }

    .prediction-character {
        font-size: 4rem;
        font-weight: bold;
    }

    .confidence {
        font-size: 1.2rem;
        color: #e2e8f0;
    }

    .top-predictions {
        background: rgba(255,255,255,0.04);
        padding: 18px;
        border-radius: 16px;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        color: #94a3b8;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# TITLE
# =========================================

st.markdown(
    """
    <h1>🪔 Ancient Tamil OCR</h1>
    <p class='subtitle'>
    Deep Learning based Ancient Tamil / Vatteluttu Glyph Recognition System
    </p>
    """,
    unsafe_allow_html=True
)

# =========================================
# LOAD MODEL
# =========================================

@st.cache_resource

def load_cnn_model():

    return load_model(
        "tamil_ocr_model.h5"
    )

model = load_cnn_model()

# =========================================
# LOAD ENCODER
# =========================================

with open(
    "label_encoder.pkl",
    "rb"
) as f:

    encoder = pickle.load(f)

# =========================================
# SETTINGS
# =========================================

IMG_SIZE = 128

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.title("ℹ️ About")

    st.write(
        "This project recognizes ancient Tamil/Vatteluttu glyphs using CNN-based OCR."
    )

    st.write("### Features")

    st.write("✅ Ancient glyph recognition")
    st.write("✅ CNN prediction")
    st.write("✅ External image handling")
    st.write("✅ Advanced preprocessing")
    st.write("✅ Top-3 predictions")

# =========================================
# CLEAR BUTTON
# =========================================

col1, col2, col3 = st.columns([1,1,1])

with col2:

    if st.button("🗑 Clear Image"):
        st.rerun()

# =========================================
# FILE UPLOADER
# =========================================

uploaded_file = st.file_uploader(
    "Upload Ancient Tamil Glyph",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff"
    ]
)

# =========================================
# PROCESS IMAGE
# =========================================

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    # =========================================
    # LAYOUT
    # =========================================

    left_col, right_col = st.columns(2)

    # =========================================
    # ORIGINAL IMAGE
    # =========================================

    with left_col:

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True
        )

        st.subheader("📷 Original Image")

        st.image(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================
    # PREPROCESSING
    # =========================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (IMG_SIZE, IMG_SIZE)
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = np.ones((2,2), np.uint8)

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        cropped = thresh[y:y+h, x:x+w]

    else:

        cropped = thresh

    canvas = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.uint8
    )

    h, w = cropped.shape

    if h > 100 or w > 100:

        scale = min(
            100 / h,
            100 / w
        )

        cropped = cv2.resize(
            cropped,
            (
                int(w * scale),
                int(h * scale)
            )
        )

        h, w = cropped.shape

    x_offset = (IMG_SIZE - w) // 2
    y_offset = (IMG_SIZE - h) // 2

    canvas[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = cropped

    # =========================================
    # NORMALIZE
    # =========================================

    img_normalized = canvas / 255.0

    img_input = img_normalized.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    # =========================================
    # PREDICT
    # =========================================

    prediction = model.predict(
        img_input,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    predicted_label = encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = np.max(prediction) * 100

    character_name = label_map.get(
        predicted_label,
        predicted_label
    )

    top3_indices = np.argsort(
        prediction[0]
    )[-3:][::-1]

    # =========================================
    # RIGHT COLUMN
    # =========================================

    with right_col:

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True
        )

        st.subheader("🧠 Processed Image")

        st.image(
            canvas,
            use_container_width=True
        )

        st.markdown(
            f"""
            <div class='prediction-box'>
                <div>Predicted Character</div>
                <div class='prediction-character'>
                    {character_name}
                </div>
                <div class='confidence'>
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "<div class='top-predictions'>",
            unsafe_allow_html=True
        )

        st.subheader("📊 Top Predictions")

        for idx in top3_indices:

            label = encoder.inverse_transform(
                [idx]
            )[0]

            character = label_map.get(
                label,
                label
            )

            score = prediction[0][idx] * 100

            st.progress(
                int(score)
            )

            st.write(
                f"{character} → {score:.2f}%"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

# =========================================
# FOOTER
# =========================================

st.markdown(
    """
    <div class='footer'>
    Developed using CNN, OpenCV, TensorFlow and Streamlit 🚀
    </div>
    """,
    unsafe_allow_html=True
)