import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# DATASET PATH
# =========================

dataset_path = "dataset"

# =========================
# IMAGE SIZE
# =========================

IMG_SIZE = 128

# =========================
# ARRAYS
# =========================

data = []
labels = []

# =========================
# LOAD DATASET
# =========================

classes = os.listdir(dataset_path)

print("Loading dataset...\n")

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing Class: {class_name}")

    for image_name in os.listdir(class_path):

        # Skip Windows system file
        if image_name == "desktop.ini":
            continue

        image_path = os.path.join(class_path, image_name)

        try:

            # Read grayscale image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize image
            img = img / 255.0

            # Store image
            data.append(img)

            # Store label
            labels.append(class_name)

        except Exception as e:

            print(f"Error loading image: {image_path}")
            print(e)

# =========================
# CONVERT TO NUMPY
# =========================

data = np.array(data)
labels = np.array(labels)

# =========================
# RESHAPE FOR CNN
# =========================

data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# =========================
# LABEL ENCODING
# =========================

encoder = LabelEncoder()

labels = encoder.fit_transform(labels)

# =========================
# SAVE LABEL ENCODER
# =========================

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=42
)

# =========================
# PRINT SHAPES
# =========================

print("\n========== DATASET SUMMARY ==========")

print("Total Images :", len(data))
print("Total Labels :", len(labels))

print("\nX_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)

print("\ny_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)

# =========================
# SAVE ARRAYS
# =========================

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nPreprocessing Completed Successfully!")

print("\nSaved Files:")
print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")
print("label_encoder.pkl")