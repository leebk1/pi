#!/usr/bin/env python3
import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2


#sensor init
from gpiozero import DistanceSensor, DigitalOutputDevice
from sensor_handler import UltrasonicSensor
from gpiozero import MotionSensor
from signal import pause

motion_ok = False

# sensor_distance = DistanceSensor(echo=24, trigger=23,max_distance=0.8)
sensor_distance = UltrasonicSensor(trigger=23, echo=24, max_distance=0.8)

led_ir = DigitalOutputDevice(17, initial_value=False)
led_ir.off()   # GPIO17 = LOW (0)


pir = MotionSensor(27)

def on_motion():
    global motion_ok
    motion_ok = True
    print("Motion detected!")

def on_no_motion():
    global motion_ok
    motion_ok = False
    print("Motion stopped")


pir.when_motion = on_motion
pir.when_no_motion = on_no_motion

print("Waiting for motion...")


# =========================
# CONFIG (match backend_face_train_verify.py style)
# =========================
# Capture (you want 800x600)
W = 800
H = 600
FPS = 30

DATASET_DIR = Path("dataset")
TMP_DIR = Path("tmp")
MODEL_PATH = "lbph_model.yml"
LABEL_MAP_PATH = "label_map.json"

# YuNet (same names/params)
YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
YUNET_INPUT_SIZE = (320, 320)
MIN_FACE_SCORE = 0.80  # same as file; if IR noface -> try 0.70 then 0.65
YUNET_NMS = 0.3
YUNET_TOPK = 1

# LBPH settings (same)
FACE_SIZE = (200, 200)
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
LBPH_CONFIDENCE_THRESH = 60.0

# Enroll/Verify
CAPTURE_COUNT = 5
CAPTURE_INTERVAL_SEC = 0.35

VERIFY_FRAMES = 10
VERIFY_INTERVAL_SEC = 0.25

# Optional preview (ssh -X/-Y)
ENABLE_PREVIEW = False


# =========================
# UTILS
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def save_gray_as_jpg(gray: np.ndarray, path: Path, quality: int = 85) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), gray, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")

def require_opencv_face():
    if not hasattr(cv2, "face"):
        raise RuntimeError("Missing cv2.face. Install: pip install opencv-contrib-python")
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("Missing FaceDetectorYN. Install: pip install opencv-contrib-python")

def save_label_map(label2name: dict[int, str]) -> None:
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label2name.items()}, f, ensure_ascii=False, indent=2)

def load_label_map() -> dict[int, str]:
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {int(k): v for k, v in d.items()}


# =========================
# Picamera2
# =========================
def open_picamera2():
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (W, H), "format": "RGB888"},
        controls={"FrameRate": FPS},
    )
    picam2.configure(cfg)
    picam2.start()
    for _ in range(5):
        _ = picam2.capture_array()
        time.sleep(0.03)
    return picam2

def capture_gray(picam2: Picamera2) -> np.ndarray:
    rgb = picam2.capture_array()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


# =========================
# YUNET FACE DETECTOR (same logic as file)
# =========================
def create_yunet_detector():
    if not os.path.exists(YUNET_MODEL):
        raise FileNotFoundError(f"Missing {YUNET_MODEL}")
    det = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        YUNET_INPUT_SIZE,
        score_threshold=MIN_FACE_SCORE,
        nms_threshold=YUNET_NMS,
        top_k=YUNET_TOPK,
    )
    return det

def crop_face(gray: np.ndarray, det) -> np.ndarray | None:
    """
    Correct YuNet usage (copied in spirit from your file):
    - resize input to detector size
    - scale bbox back to original image
    - pad 0.18
    """
    H0, W0 = gray.shape[:2]

    # YuNet expects BGR
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    input_w, input_h = YUNET_INPUT_SIZE
    det.setInputSize((input_w, input_h))

    resized = cv2.resize(bgr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    _, faces = det.detect(resized)
    if faces is None or len(faces) == 0:
        return None

    # Face bbox in resized coordinates
    x, y, w, h, score = faces[0][0], faces[0][1], faces[0][2], faces[0][3], float(faces[0][4])

    # Scale back
    sx = W0 / input_w
    sy = H0 / input_h
    x = int(x * sx)
    y = int(y * sy)
    w = int(w * sx)
    h = int(h * sy)

    pad = int(0.18 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W0, x + w + pad)
    y1 = min(H0, y + h + pad)

    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    return roi


# =========================
# PREPROCESS + "SYNTH IR" AUGMENT (same as file)
# =========================
def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess_for_lbph(gray: np.ndarray) -> np.ndarray:
    """
    Same as your file:
    - Resize to FACE_SIZE
    - DoG (sigma 1.0 and 2.0)
    - Normalize
    - CLAHE clipLimit=2.5
    """
    img = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    g1 = cv2.GaussianBlur(img, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(img, (0, 0), 2.0)
    dog = cv2.subtract(g1, g2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(dog)
    return out

def synth_ir_variants(face_roi_gray: np.ndarray) -> list[np.ndarray]:
    outs = []
    outs.append(preprocess_for_lbph(face_roi_gray))
    outs.append(preprocess_for_lbph(apply_gamma(face_roi_gray, 1.4)))
    outs.append(preprocess_for_lbph(apply_gamma(face_roi_gray, 1.8)))
    for alpha, beta in [(1.4, -15), (1.8, -35)]:
        v = cv2.convertScaleAbs(face_roi_gray, alpha=alpha, beta=beta)
        outs.append(preprocess_for_lbph(v))
    img = cv2.resize(face_roi_gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    outs.append(clahe_strong.apply(img))
    img = cv2.resize(face_roi_gray, FACE_SIZE, interpolation=cv2.INTER_AREA)
    noise = np.random.normal(0, 7, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    outs.append(preprocess_for_lbph(noisy))
    return outs


# =========================
# LBPH MODEL IO (same as file)
# =========================
def create_recognizer():
    require_opencv_face()
    return cv2.face.LBPHFaceRecognizer_create(
        radius=LBPH_RADIUS,
        neighbors=LBPH_NEIGHBORS,
        grid_x=LBPH_GRID_X,
        grid_y=LBPH_GRID_Y,
    )

def load_recognizer():
    require_opencv_face()
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_PATH)
    return rec


# =========================
# DATASET + TRAIN (same logic as file)
# =========================
def list_user_dirs(dataset_dir: Path) -> list[str]:
    if not dataset_dir.is_dir():
        return []
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])

def load_dataset_for_lbph_with_face_roi_aug(dataset_dir: Path, det):
    images = []
    labels = []

    names = list_user_dirs(dataset_dir)
    if not names:
        raise RuntimeError("No user folders found under dataset/")

    name2label = {name: i for i, name in enumerate(names)}
    label2name = {i: name for name, i in name2label.items()}

    for name in names:
        person_dir = dataset_dir / name
        for fp in person_dir.glob("*.jpg"):
            gray = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            face = crop_face(gray, det)
            if face is None:
                continue

            for v in synth_ir_variants(face):
                images.append(v)
                labels.append(name2label[name])

    if not images:
        raise RuntimeError("No usable face crops found. Ensure face is visible in dataset images.")

    return images, np.array(labels, dtype=np.int32), label2name

def train_lbph_from_dataset(dataset_dir: Path, det):
    images, labels, label2name = load_dataset_for_lbph_with_face_roi_aug(dataset_dir, det)
    rec = create_recognizer()
    rec.train(images, labels)
    rec.write(MODEL_PATH)
    save_label_map(label2name)
    return rec, label2name


def predict_lbph(recognizer, label2name: dict[int, str], gray: np.ndarray, det):
    face = crop_face(gray, det)
    if face is None:
        return "noface", 1e9

    img = preprocess_for_lbph(face)
    label, conf = recognizer.predict(img)
    label = int(label)
    conf = float(conf)

    name = label2name.get(label, "unknown")
    if conf > LBPH_CONFIDENCE_THRESH:
        return "unknown", conf
    return name, conf

LED_ON = False
def led_on(opt_on=False,is_check=False):

    global LED_ON,led_ir
    if is_check:#only check, not set
        if opt_on==LED_ON:
            return
    if opt_on:
        LED_ON = True
        led_ir.on()
    else:
        LED_ON = False
        led_ir.off()



# =========================
# MODES
# =========================
def enroll_mode(picam2: Picamera2):
    det = create_yunet_detector()

    user_name = input("Enter NEW user name (folder under dataset/): ").strip()
    if not user_name:
        print("[ERROR] Empty name.")
        return

    user_dir = DATASET_DIR / user_name
    ensure_dir(user_dir)

    # Nếu đã có ảnh -> dừng
    if any(user_dir.glob("*.jpg")):
        print(f"[INFO] '{user_name}' already exists. Exiting.")
        return

    print(f"[INFO] Capturing {CAPTURE_COUNT} frames (GRAY) @ {W}x{H} into {user_dir} ...")
    saved = 0
    for i in range(CAPTURE_COUNT):
        gray = capture_gray(picam2)

        # Debug detect right now
        face = crop_face(gray, det)
        if face is None:
            print(f"[WARN] noface on capture {i+1}/{CAPTURE_COUNT} (try lower MIN_FACE_SCORE or adjust IR).")

        out_path = user_dir / f"{user_name}_{ts()}_{i:03d}.jpg"
        save_gray_as_jpg(gray, out_path, quality=85)
        saved += 1
        print(f"[INFO] saved {out_path}")

        if ENABLE_PREVIEW:
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imshow("enroll_gray", disp)
            cv2.waitKey(1)

        time.sleep(CAPTURE_INTERVAL_SEC)

    if saved < 5:
        print("[ERROR] Too few samples saved. Fix framing/lighting and retry.")
        return

    print("[INFO] Training LBPH using YuNet face ROI + illumination-robust preprocess + IR-like augmentation ...")
    rec, label2name = train_lbph_from_dataset(DATASET_DIR, det)
    print(f"[INFO] Model saved: {MODEL_PATH}")
    print(f"[INFO] Label map saved: {LABEL_MAP_PATH}")


def detect_mode(picam2: Picamera2):
    det = create_yunet_detector()

    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH)):
        print("[ERROR] Missing model files. Enroll/train first (option 1).")
        return

    rec = load_recognizer()
    label2name = load_label_map()

    ensure_dir(TMP_DIR)

    print(f"[INFO] Verifying {VERIFY_FRAMES} frames from Picamera2 @ {W}x{H} ...")
    votes = {}
    best_conf = {}
    ok_frames = 0
    i=0
    while True:
        if not motion_ok:
            time.sleep(1)
            has_object = False
            led_on(False,True)
            continue
        if not has_object:
            distance_human = sensor_distance.read()
            if distance_human is not None and distance_human < 0.8:
                print(f"distance ok {distance_human}m")
                has_object = True
            else:
                time.sleep(0.5)
                continue
        if not LED_ON:
            led_on(opt_on=True)
    # for i in range(VERIFY_FRAMES):
        gray = capture_gray(picam2)
        name, conf = predict_lbph(rec, label2name, gray, det)
        if name=="noface":
            print("noface detected")
            time.sleep(VERIFY_INTERVAL_SEC)
            continue
        ok_frames += 1
        votes[name] = votes.get(name, 0) + 1
        best_conf[name] = min(best_conf.get(name, 1e9), conf)

        tmp_path = TMP_DIR / f"{name}_{ts()}.jpg"
        save_gray_as_jpg(gray, tmp_path, quality=85)
        print(f"[INFO] frame {i+1}/{VERIFY_FRAMES}: pred={name} conf={conf:.2f} saved={tmp_path}")

        break
        time.sleep(VERIFY_INTERVAL_SEC)

    if ok_frames == 0:
        print("[ERROR] No frames captured successfully.")
        return
    led_on(False)
    winner = sorted(votes.items(), key=lambda kv: (-kv[1], best_conf.get(kv[0], 1e9)))[0][0]
    result = {
        "timestamp": datetime.now().isoformat(),
        "mode": "verify",
        "frame": {"width": W, "height": H},
        "votes": votes,
        "best_conf": {k: float(v) for k, v in best_conf.items()},
        "winner": winner,
        "threshold": LBPH_CONFIDENCE_THRESH,
        "frames_ok": ok_frames,
        "frames_requested": VERIFY_FRAMES,
        "tmp_dir": str(TMP_DIR),
    }
    print("[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    time.sleep(3)


def main():
    require_opencv_face()
    ensure_dir(DATASET_DIR)
    ensure_dir(TMP_DIR)

    print(f"[INFO] OpenCV={cv2.__version__}")
    print(f"[INFO] Capture={W}x{H}@{FPS}")
    picam2 = open_picamera2()

    try:
        print("==== MENU ====")
        print("1) Enroll new user (capture + train)")
        print("2) Verify (who is in front of camera) + save frames to tmp/")
        choice = input("Select option (1/2): ").strip()

        if choice == "1":
            enroll_mode(picam2)
        elif choice == "2":
            detect_mode(picam2)
        else:
            print("[ERROR] Invalid option.")
    finally:
        picam2.stop()
        if ENABLE_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
