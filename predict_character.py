import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================

print("Loading trained model...\n")

model = load_model("tamil_ocr_model.h5")

print("Model loaded successfully!\n")

# =========================
# LOAD LABEL ENCODER
# =========================

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# =========================
# SETTINGS
# =========================

IMG_SIZE = 128

# =========================
# IMAGE PATH
# =========================

image_path = "test_images/test3.tiff"

# =========================
# LOAD IMAGE
# =========================

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

# =========================
# CONVERT TO GRAYSCALE
# =========================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# RESIZE IMAGE
# =========================

gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

# =========================
# DENOISE
# =========================

gray = cv2.GaussianBlur(gray, (5, 5), 0)

# =========================
# ADAPTIVE THRESHOLDING
# =========================

thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# =========================
# MORPHOLOGICAL CLEANING
# =========================

kernel = np.ones((2, 2), np.uint8)

thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)

# =========================
# FIND CONTOURS
# =========================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# =========================
# CROP MAIN GLYPH
# =========================

if contours:

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped = thresh[y:y+h, x:x+w]

else:

    cropped = thresh

# =========================
# CENTER GLYPH ON CANVAS
# =========================

canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

h, w = cropped.shape

# Scale if too large
if h > 100 or w > 100:

    scale = min(100 / h, 100 / w)

    cropped = cv2.resize(
        cropped,
        (
            int(w * scale),
            int(h * scale)
        )
    )

    h, w = cropped.shape

# Center image
x_offset = (IMG_SIZE - w) // 2
y_offset = (IMG_SIZE - h) // 2

canvas[
    y_offset:y_offset+h,
    x_offset:x_offset+w
] = cropped

# =========================
# NORMALIZE
# =========================

img_normalized = canvas / 255.0

# =========================
# RESHAPE FOR CNN
# =========================

img_input = img_normalized.reshape(
    1,
    IMG_SIZE,
    IMG_SIZE,
    1
)

# =========================
# PREDICT
# =========================

prediction = model.predict(img_input)

predicted_index = np.argmax(prediction)

predicted_label = encoder.inverse_transform(
    [predicted_index]
)[0]

confidence = np.max(prediction) * 100

# =========================
# TOP 3 PREDICTIONS
# =========================

top3_indices = np.argsort(
    prediction[0]
)[-3:][::-1]

print("\n========== PREDICTION RESULTS ==========\n")

print(f"Predicted Class : {predicted_label}")

print(f"Confidence Score: {confidence:.2f}%\n")

print("Top 3 Predictions:\n")

for idx in top3_indices:

    label = encoder.inverse_transform([idx])[0]

    score = prediction[0][idx] * 100

    print(f"Class {label} --> {score:.2f}%")

# =========================
# DISPLAY RESULTS
# =========================

plt.figure(figsize=(12, 4))

# Original
plt.subplot(1, 3, 1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.title("Original Image")

plt.axis("off")

# Threshold
plt.subplot(1, 3, 2)

plt.imshow(thresh, cmap='gray')

plt.title("Thresholded")

plt.axis("off")

# Final Processed
plt.subplot(1, 3, 3)

plt.imshow(canvas, cmap='gray')

plt.title(f"Prediction: {predicted_label}")

plt.axis("off")

plt.show()

print("\nPrediction Completed Successfully!")