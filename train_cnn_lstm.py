import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

BASE_PATH       = "D:/session-06-26/Drowsiness_Detection"
DATASET_PATH    = os.path.join(BASE_PATH, "train")
CNN_MODEL_PATH  = os.path.join(BASE_PATH, "models", "cnn_eye_model.h5")
LSTM_MODEL_PATH = os.path.join(BASE_PATH, "models", "cnn_lstm_model.h5")

SEQ_LEN  = 10
IMG_SIZE = 24

def load_data():
    data, labels = [], []

    for img_name in os.listdir(os.path.join(DATASET_PATH, "Open")):
        img = cv2.imread(os.path.join(DATASET_PATH, "Open", img_name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(1)

    for img_name in os.listdir(os.path.join(DATASET_PATH, "Closed")):
        img = cv2.imread(os.path.join(DATASET_PATH, "Closed", img_name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            data.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            labels.append(0)

    data   = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    data   = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    total   = len(data)
    indices = np.random.permutation(total)
    split   = int(total * 0.8)

    return (data[indices[:split]], data[indices[split:]],
            labels[indices[:split]], labels[indices[split:]])

def create_sequences(features, labels, seq_len=SEQ_LEN):
    sequences, seq_labels = [], []
    for i in range(len(features) - seq_len):
        sequences.append(features[i:i+seq_len])
        label = 0 if np.mean(labels[i:i+seq_len]) < 0.5 else 1
        seq_labels.append(label)
    return np.array(sequences), np.array(seq_labels)

def train():
    # Load CNN model aur dummy predict se build karo
    cnn_model = load_model(CNN_MODEL_PATH)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32")
    cnn_model.predict(dummy, verbose=0)
    print("[INFO] CNN model loaded")

    # Feature extractor — Dense(128) layer se pehle ka output
    feature_extractor = Model(
        inputs=cnn_model.inputs,
        outputs=cnn_model.layers[-3].output
    )
    feature_extractor.trainable = False

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    print(f"[INFO] Data loaded: {len(X_train)} train, {len(X_test)} test")

    # Extract CNN features
    print("[INFO] Extracting CNN features...")
    train_features = feature_extractor.predict(X_train, batch_size=32, verbose=0)
    test_features  = feature_extractor.predict(X_test,  batch_size=32, verbose=0)
    print(f"[INFO] Feature shape: {train_features.shape}")

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(train_features, y_train)
    X_test_seq,  y_test_seq  = create_sequences(test_features,  y_test)
    print(f"[INFO] Train sequences: {X_train_seq.shape}")
    print(f"[INFO] Test sequences : {X_test_seq.shape}")

    # LSTM model
    feature_dim = train_features.shape[1]
    lstm_model = Sequential([
        Input(shape=(SEQ_LEN, feature_dim)),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    lstm_model.summary()

    callbacks = [
        ModelCheckpoint(LSTM_MODEL_PATH, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=10, restore_best_weights=True)
    ]

    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=callbacks
    )

    loss, acc = lstm_model.evaluate(X_test_seq, y_test_seq)
    print(f"\n[DONE] CNN-LSTM Accuracy: {acc*100:.2f}%")
    print(f"[DONE] Model saved: {LSTM_MODEL_PATH}")

if __name__ == "__main__":
    train()