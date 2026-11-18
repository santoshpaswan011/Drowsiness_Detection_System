import os
import cv2
import numpy as np

BASE_PATH = "D:/session-06-26/Drowsiness_Detection"
DATASET_PATH = os.path.join(BASE_PATH, "train")

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

    # Manual train/test split
    total = len(data)
    indices = np.random.permutation(total)
    split = int(total * 0.8)

    train_idx = indices[:split]
    test_idx  = indices[split:]

    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f" Total images : {total}")
    print(f" Train samples: {len(X_train)}")
    print(f" Test samples : {len(X_test)}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_data()