import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pygame
import os
from collections import deque
from tensorflow.keras.models import load_model, Model

# Paths
BASE_PATH        = "D:/session-06-26/Drowsiness_Detection"
CNN_MODEL_PATH   = os.path.join(BASE_PATH, "models", "cnn_eye_model.h5")
LSTM_MODEL_PATH  = os.path.join(BASE_PATH, "models", "cnn_lstm_model.h5")
ALARM_PATH       = os.path.join(BASE_PATH, "alarm.wav")
FACE_MODEL_PATH  = os.path.join(BASE_PATH, "face_landmarker.task")

IMG_SIZE         = 24
SEQ_LEN          = 10
DROWSY_THRESHOLD = 0.5

# Alarm setup
pygame.mixer.init()
alarm_sound   = pygame.mixer.Sound(ALARM_PATH)
alarm_playing = False

# Load models
print("[INFO] Loading models...")
cnn_model  = load_model(CNN_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype="float32")
cnn_model.predict(dummy, verbose=0)
feature_extractor = Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.layers[-3].output
)
print("[INFO] Models loaded")

# Eye landmark indices
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# MediaPipe Face Landmarker
base_options    = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options    = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
print("[INFO] Face landmarker loaded")

def get_eye_roi(frame, landmarks, eye_indices, h, w):
    points   = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1 = max(min(x_coords) - 10, 0)
    x2 = min(max(x_coords) + 10, w)
    y1 = max(min(y_coords) - 10, 0)
    y2 = min(max(y_coords) + 10, h)
    return frame[y1:y2, x1:x2]

def preprocess_eye(eye_roi):
    if eye_roi is None or eye_roi.size == 0:
        return None
    gray    = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype("float32") / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def extract_feature(eye_input):
    feat = feature_extractor.predict(eye_input, verbose=0)
    return feat[0]

# Feature buffer
feature_buffer = deque(maxlen=SEQ_LEN)

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for _ in range(5):
    cap.read()

print("[INFO] Starting detection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = face_landmarker.detect(mp_image)

    status = "No Face"
    color  = (0, 255, 255)

    if detection.face_landmarks:
        landmarks = detection.face_landmarks[0]

        left_roi  = get_eye_roi(frame, landmarks, LEFT_EYE,  h, w)
        right_roi = get_eye_roi(frame, landmarks, RIGHT_EYE, h, w)

        left_input  = preprocess_eye(left_roi)
        right_input = preprocess_eye(right_roi)

        if left_input is not None and right_input is not None:
            left_feat  = extract_feature(left_input)
            right_feat = extract_feature(right_input)
            avg_feat   = (left_feat + right_feat) / 2.0

            feature_buffer.append(avg_feat)

            if len(feature_buffer) == SEQ_LEN:
                seq         = np.array(list(feature_buffer)).reshape(1, SEQ_LEN, 128)
                drowsy_prob = lstm_model.predict(seq, verbose=0)[0][0]

                # prob < 0.5 = closed = DROWSY
                # prob > 0.5 = open   = ALERT
                if drowsy_prob < DROWSY_THRESHOLD:
                    status = "DROWSY! ({:.0f}%)".format((1 - drowsy_prob) * 100)
                    color  = (0, 0, 255)
                    if not alarm_playing:
                        alarm_sound.play(-1)
                        alarm_playing = True
                else:
                    status = "ALERT ({:.0f}%)".format(drowsy_prob * 100)
                    color  = (0, 255, 0)
                    if alarm_playing:
                        alarm_sound.stop()
                        alarm_playing = False
            else:
                status = "Collecting {}/{}".format(len(feature_buffer), SEQ_LEN)
                color  = (255, 255, 0)

    # Overlay
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, status, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

alarm_sound.stop()
cap.release()
cv2.destroyAllWindows()
print("[INFO] Stopped.")