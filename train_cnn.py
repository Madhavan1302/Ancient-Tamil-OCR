import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# LOAD DATASET
# =========================

print("Loading processed dataset...\n")

X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# =========================
# NUMBER OF CLASSES
# =========================

num_classes = len(np.unique(y_train))

print("Number of Classes:", num_classes)

# =========================
# ONE HOT ENCODING
# =========================

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# =========================
# BUILD CNN MODEL
# =========================

model = Sequential()

# First Convolution Block
model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 1)
    )
)

model.add(MaxPooling2D((2, 2)))

# Second Convolution Block
model.add(
    Conv2D(
        64,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2, 2)))

# Third Convolution Block
model.add(
    Conv2D(
        128,
        (3, 3),
        activation='relu'
    )
)

model.add(MaxPooling2D((2, 2)))

# Flatten Layer
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(num_classes, activation='softmax'))

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# MODEL SUMMARY
# =========================

print("\nCNN MODEL SUMMARY:\n")

model.summary()

# =========================
# EARLY STOPPING
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

# =========================
# EVALUATE MODEL
# =========================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

model.save("tamil_ocr_model.h5")

print("\nModel Saved Successfully!")

# =========================
# PLOT ACCURACY GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()

# =========================
# PLOT LOSS GRAPH
# =========================

plt.figure(figsize=(10, 5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()

print("\nCNN Training Completed Successfully!")