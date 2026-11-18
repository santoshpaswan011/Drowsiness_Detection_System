import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

BASE_PATH = "D:/session-06-26/Drowsiness_Detection"
DATASET_PATH = os.path.join(BASE_PATH, "train")
MODEL_PATH = os.path.join(BASE_PATH, "models", "cnn_eye_model.h5")

IMG_SIZE = 24

def load_data():
    data = []
    labels = []

    open_path = os.path.join(DATASET_PATH, "Open")
    for img_name in os.listdir(open_path):
        img_path = os.path.join(open_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(1)

    closed_path = os.path.join(DATASET_PATH, "Closed")
    for img_name in os.listdir(closed_path):
        img_path = os.path.join(closed_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(0)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    total = len(data)
    indices = np.random.permutation(total)
    split = int(total * 0.8)

    train_idx = indices[:split]
    test_idx  = indices[split:]

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"[OK] Train samples: {len(X_train)}")
    print(f"[OK] Test samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    X_train, X_test, y_train, y_test = load_data()

    model = build_cnn()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n[DONE] CNN Test Accuracy: {acc*100:.2f}%")

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('CNN Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('CNN Loss')
    plt.legend()

    plt.savefig(os.path.join(BASE_PATH, "models", "cnn_training_plot.png"))
    plt.show()
    print(f"[DONE] Model saved: {MODEL_PATH}")

if __name__ == "__main__":
    train()