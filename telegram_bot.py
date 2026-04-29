import requests
from requests.auth import HTTPDigestAuth
import time
import json
import os
import io
import threading
import numpy as np
from PIL import Image
from gpiozero import OutputDevice
from datetime import datetime

# --- Config ---
TOKEN = "6534373808:AAGEN9S-lES2LA8efbSxLXxYWvsSLgLAGrY"
OWNER_ID = 430228336
TX_PIN = 17
CAM_URL = "http://192.168.1.102/ISAPI/Streaming/channels/101/picture"
CAM_USER = "admin"
CAM_PASS = "Viettel@2026"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_FILE = os.path.join(BASE_DIR, "admins.json")
OPEN_COOLDOWN = 20  # giây
LOG_FILE = os.path.join(BASE_DIR, "activity.log")
TRAIN_DIR = os.path.join(BASE_DIR, "train_data")
OPEN_DIR = os.path.join(TRAIN_DIR, "open")
CLOSE_DIR = os.path.join(TRAIN_DIR, "close")
MODEL_FILE = os.path.join(BASE_DIR, "door_model.pkl")
IMG_SIZE = 64
NIGHT_ALERT_HOUR = 20  # 8h tối
NIGHT_CHECK_INTERVAL = 15 * 60  # 15 phút
last_open_time = 0
training_lock = threading.Lock()


def load_admins():
    """Load admins dict {id_str: display_name} từ file."""
    try:
        with open(ADMIN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Tương thích format cũ (list)
        if isinstance(data, list):
            return {uid: "" for uid in data}
        # Format mới: {"430228336": "Nguyen Van A"}
        return {int(k): v for k, v in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {OWNER_ID: "Owner"}


def save_admins():
    with open(ADMIN_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in ADMIN_DB.items()}, f, ensure_ascii=False, indent=2)


ADMIN_DB = load_admins()  # {user_id: display_name}


def get_display_name(from_obj):
    """Lấy tên hiển thị từ object 'from' của Telegram message."""
    parts = []
    if from_obj.get("first_name"):
        parts.append(from_obj["first_name"])
    if from_obj.get("last_name"):
        parts.append(from_obj["last_name"])
    name = " ".join(parts)
    if from_obj.get("username"):
        name += " (@{})".format(from_obj["username"])
    return name or str(from_obj.get("id", ""))


def update_admin_name(user_id, from_obj):
    """Cập nhật tên hiển thị của admin nếu có thay đổi."""
    if user_id in ADMIN_DB:
        name = get_display_name(from_obj)
        if ADMIN_DB[user_id] != name:
            ADMIN_DB[user_id] = name
            save_admins()


def write_log(user_id, action):
    name = ADMIN_DB.get(user_id, str(user_id))
    line = "{} | {} ({}) | {}\n".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), name, user_id, action)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def read_log(n=20):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return lines[-n:]
    except FileNotFoundError:
        return []


# --- zero.sub data embedded ---
ZERO_SUB_RAW = "3733 -16694 197 -930 99 -166 131 -570 133 -66 697 -132 233 -134 99 -134 1363 -17254 167 -132 199 -1094 197 -64 165 -198 723 -728 833 -98 1195 -17570 263 -166 97 -732 825 -266 199 -134 165 -200 68679 -18362 67 -98 67 -98 133 -132 263 -460 263 -396 163 -196 97 -66 393 -132 299 -102 10677 -16686 331 -264 65 -168 99 -498 65 -434 65 -166 793 -98 163 -164 325 -132 65 -98 65 -132 629 -68 41905 -66 9995 -102 45413 -16424 227 -626 65 -268 99 -398 99 -132 691 -100 393 -130 14321 -16550 99 -198 67 -366 329 -556 395 -164 595 -298 199 -198 1425 -13146 131 -66 297 -236 363 -200 265 -66 65 -166 1097 -66 363 -100 8279 -16674 523 -1152 225 -1198 731 -654 849 -608 369 -1082 353 -1068 897 -554 887 -574 893 -522 429 -1038 425 -1014 923 -540 425 -1012 431 -1006 455 -1012 905 -542 427 -1008 459 -976 955 -490 471 -1002 451 -978 473 -980 949 -488 471 -984 491 -976 461 -980 949 -488 965 -9784 469 -976 457 -986 985 -452 985 -466 485 -974 487 -982 937 -488 981 -478 943 -490 479 -980 455 -986 953 -486 473 -978 479 -944 479 -970 973 -488 483 -946 483 -954 955 -484 501 -948 483 -976 475 -974 941 -486 485 -948 483 -954 495 -978 955 -478 949 -9722 495 -964 481 -948 971 -482 953 -480 495 -942 497 -946 981 -486 953 -482 943 -482 473 -978 485 -974 957 -474 459 -972 493 -972 471 -942 961 -484 501 -942 483 -976 965 -478 489 -940 471 -974 493 -940 959 -506 453 -972 471 -976 491 -938 965 -482 957 -142014 199 -1664 429 -230 1559 -198 36015 -17100 363 -464 131 -100 101 -864 265 -100 265 -266 399 -166 263 -132 81653 -17210 99 -232 131 -336 65 -232 99 -232 99 -266 167 -98 499 -98 165 -134 199 -68 427 -132 1661 -13868 165 -1694 163 -890 97 -232 195 -196 659 -398 559 -132 23749 -17258 231 -496 131 -166 65 -428 197 -198 229 -66 227 -64 295 -98 229 -98 261 -98 655 -68 693 -100 8605 -16832 131 -2000 331 -264 295 -894 199 -134 695 -100 3681 -16646 65 -398 99 -200 65 -134 163 -530 463 -166 36001 -17172 397 -1152 329 -98 919 -198 331 -398 43211 -16500 167 -1098 65 -66 165 -1552 397 -66 297 -232 1725 -16752 197 -164 67 -200 99 -132 365 -922 427 -162 293 -130 65 -528 2495 -66"


def _print(msg):
    print("{}  -  {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


def api_request(method, **kwargs):
    """Helper gọi Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TOKEN}/{method}"
    try:
        resp = requests.post(url, json=kwargs, timeout=35)
        return resp.json()
    except Exception as e:
        _print("API error [{}]: {}".format(method, e))
        return {}


def set_bot_commands():
    """Đăng ký menu commands cho bot."""
    commands = [
        {"command": "view", "description": "Chụp ảnh từ camera"},
        {"command": "open", "description": "Mở cửa"},
        {"command": "status", "description": "Kiểm tra cửa đang mở hay đóng"},
        {"command": "trainopen", "description": "Train ảnh cửa mở (owner)"},
        {"command": "trainclose", "description": "Train ảnh cửa đóng (owner)"},
        {"command": "addadmin", "description": "Thêm admin (reply hoặc /addadmin <id>)"},
        {"command": "removeadmin", "description": "Xóa admin (/removeadmin <id>)"},
        {"command": "log", "description": "Xem lịch sử hoạt động (owner)"},
        {"command": "listadmin", "description": "Xem danh sách admin"},
        {"command": "myid", "description": "Xem chat ID của bạn"},
        {"command": "start", "description": "Hiển thị hướng dẫn"},
    ]
    result = api_request("setMyCommands", commands=commands)
    _print("Menu commands registered: {}".format(result.get("ok")))


def send_message(chat_id, text, reply_markup=None):
    kwargs = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_markup:
        kwargs["reply_markup"] = reply_markup
    api_request("sendMessage", **kwargs)


def get_main_keyboard():
    """Reply keyboard luôn hiển thị trong chat."""
    return {
        "keyboard": [
            [{"text": "/open"}, {"text": "/view"}, {"text": "/status"}],
            [{"text": "/listadmin"}, {"text": "/myid"}],
        ],
        "resize_keyboard": True,
        "is_persistent": True,
    }


def send_photo(chat_id, photo_bytes, caption=None):
    """Gửi ảnh qua Telegram."""
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    try:
        requests.post(url, data=data, files={"photo": ("snapshot.jpg", photo_bytes, "image/jpeg")}, timeout=30)
    except Exception as e:
        _print("Send photo error: {}".format(e))


def capture_snapshot():
    """Chụp ảnh từ camera Hikvision."""
    try:
        resp = requests.get(CAM_URL, auth=HTTPDigestAuth(CAM_USER, CAM_PASS), timeout=10)
        if resp.status_code == 200:
            return resp.content
        _print("Camera HTTP {}: {}".format(resp.status_code, resp.text[:100]))
    except Exception as e:
        _print("Camera error: {}".format(e))
    return None


# --- Door detection (train / predict) ---

def preprocess_image(image_bytes):
    """Chuyển ảnh thành grayscale, cắt nửa trái, resize 64x64, flatten."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale (xử lý cả RGB lẫn IR)
    w, h = img.size
    img = img.crop((0, 0, w // 2, h))  # chỉ lấy nửa trái
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32).flatten() / 255.0


def load_training_data():
    """Load tất cả ảnh train từ thư mục open/ và close/."""
    X, y = [], []
    for label, folder in [(1, OPEN_DIR), (0, CLOSE_DIR)]:
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            try:
                with open(fpath, "rb") as f:
                    data = preprocess_image(f.read())
                X.append(data)
                y.append(label)
            except Exception as e:
                _print("Lỗi đọc ảnh train {}: {}".format(fpath, e))
    return np.array(X), np.array(y)


def train_model():
    """Huấn luyện MLP classifier từ ảnh đã thu thập."""
    from sklearn.neural_network import MLPClassifier
    import joblib

    X, y = load_training_data()

    n_open = int(np.sum(y == 1))
    n_close = int(np.sum(y == 0))
    _print("Train data: {} ảnh mở, {} ảnh đóng".format(n_open, n_close))

    if n_open < 2 or n_close < 2:
        return False, "Chưa đủ dữ liệu (cần ít nhất 2 ảnh mỗi loại). Hiện có: {} mở, {} đóng.".format(n_open, n_close)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    score = model.score(X, y)
    _print("Model trained. Accuracy trên tập train: {:.1f}%".format(score * 100))
    return True, "Huấn luyện hoàn tất! Accuracy: {:.1f}% ({} ảnh mở, {} ảnh đóng)".format(score * 100, n_open, n_close)


def predict_door(image_bytes):
    """Dự đoán cửa mở hay đóng từ ảnh."""
    import joblib

    if not os.path.exists(MODEL_FILE):
        return None, "Chưa có model. Hãy dùng /trainopen và /trainclose trước."

    model = joblib.load(MODEL_FILE)
    data = preprocess_image(image_bytes).reshape(1, -1)
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0]
    confidence = max(proba) * 100
    status = "MỞ" if pred == 1 else "ĐÓNG"
    return status, "Cửa đang: <b>{}</b> (độ tin cậy: {:.1f}%)".format(status, confidence)


def has_model():
    return os.path.exists(MODEL_FILE)


def quick_predict(image_bytes):
    """Predict nhanh, trả về 'MỞ'/'ĐÓNG' hoặc None nếu chưa có model."""
    if not has_model():
        return None
    status, _ = predict_door(image_bytes)
    return status


def _open_door_thread(chat_id, user_id, kb):
    """Thread: chụp trước -> mở cửa -> chờ 15s -> chụp lại -> so sánh trạng thái."""
    global last_open_time

    # Bước 1: chụp ảnh trước khi mở
    before_status = None
    if has_model():
        photo_before = capture_snapshot()
        if photo_before:
            before_status = quick_predict(photo_before)
            send_photo(chat_id, photo_before,
                       caption="Trước khi mở - Trạng thái: {}".format(before_status or "N/A"))
            _print("Trước khi mở: {}".format(before_status))

    # Bước 2: phát tín hiệu mở cửa
    send_message(chat_id, "Đang phát tín hiệu mở cửa...", kb)
    success = send_trigger()
    if not success:
        send_message(chat_id, "Lỗi khi phát tín hiệu!", kb)
        return

    last_open_time = time.time()
    write_log(user_id, "Mở cửa")
    send_message(chat_id, "Đã gửi tín hiệu mở cửa. Chờ 20s để xác nhận...", kb)

    # Bước 3: chờ 20s rồi chụp lại
    if has_model():
        time.sleep(20)
        photo_after = capture_snapshot()
        if photo_after:
            after_status = quick_predict(photo_after)
            send_photo(chat_id, photo_after,
                       caption="Sau khi mở - Trạng thái: {}".format(after_status or "N/A"))
            _print("Sau khi mở: {}".format(after_status))

            # So sánh trạng thái
            if before_status and after_status:
                if before_status == after_status:
                    send_message(chat_id,
                                 "⚠ Trạng thái không đổi (vẫn <b>{}</b>). Có thể cửa mở bị lỗi!".format(after_status), kb)
                    write_log(user_id, "Cảnh báo: cửa mở lỗi (trạng thái không đổi)")
                else:
                    send_message(chat_id,
                                 "Xác nhận: cửa đã chuyển từ <b>{}</b> sang <b>{}</b>.".format(before_status, after_status), kb)
            elif after_status:
                send_message(chat_id, "Trạng thái hiện tại: <b>{}</b>".format(after_status), kb)
        else:
            send_message(chat_id, "Không thể chụp ảnh xác nhận.", kb)
    else:
        send_message(chat_id, "Đã mở cửa thành công! (Chưa có model để xác nhận trạng thái)", kb)


def _night_watch_thread():
    """Thread chạy nền: sau 20h, mỗi 15p check cửa, nếu MỞ thì cảnh báo owner."""
    _print("Night watch thread started.")
    while True:
        now = datetime.now()
        hour = now.hour

        # Chỉ check từ 20h đến 6h sáng
        if hour >= NIGHT_ALERT_HOUR or hour < 6:
            if has_model():
                photo = capture_snapshot()
                if photo:
                    status = quick_predict(photo)
                    if status == "MỞ":
                        _print("Night alert: cửa đang MỞ lúc {}".format(now.strftime('%H:%M')))
                        send_message(OWNER_ID,
                                     "🚨 <b>CẢNH BÁO:</b> Cửa đang <b>MỞ</b> lúc {}!\nHãy kiểm tra ngay.".format(
                                         now.strftime('%H:%M:%S')))
                        send_photo(OWNER_ID, photo,
                                   caption="Cảnh báo ban đêm - Cửa MỞ lúc {}".format(now.strftime('%H:%M:%S')))
                        write_log(0, "Cảnh báo tự động: cửa MỞ ban đêm")

        time.sleep(NIGHT_CHECK_INTERVAL)


def _train_capture_thread(chat_id, train_dir, label_name, kb):
    """Thread chụp 3 ảnh cách nhau 17s và train lại model."""
    os.makedirs(train_dir, exist_ok=True)
    captured = 0

    for i in range(3):
        if i > 0:
            time.sleep(17)
        photo = capture_snapshot()
        if photo:
            captured += 1
            fname = datetime.now().strftime('%Y%m%d_%H%M%S') + ".jpg"
            fpath = os.path.join(train_dir, fname)
            with open(fpath, "wb") as f:
                f.write(photo)
            send_message(chat_id, "Đã chụp ảnh {} ({}/3)".format(label_name, i + 1), kb)
            send_photo(chat_id, photo, caption="{} - ảnh {}/3".format(label_name, i + 1))
        else:
            send_message(chat_id, "Lỗi chụp ảnh lần {}/3".format(i + 1), kb)

    if captured == 0:
        send_message(chat_id, "Không chụp được ảnh nào!", kb)
        training_lock.release()
        return

    # Tự động train lại nếu đủ data cả 2 lớp
    n_open = len(os.listdir(OPEN_DIR)) if os.path.isdir(OPEN_DIR) else 0
    n_close = len(os.listdir(CLOSE_DIR)) if os.path.isdir(CLOSE_DIR) else 0

    if n_open >= 2 and n_close >= 2:
        send_message(chat_id, "Đang huấn luyện model...", kb)
        success, msg = train_model()
        send_message(chat_id, msg, kb)
    else:
        send_message(chat_id, "Đã lưu {} ảnh. Cần thêm dữ liệu để train (mở: {}, đóng: {}).".format(
            captured, n_open, n_close), kb)

    training_lock.release()


def send_trigger():
    """Replay RF signal from embedded zero.sub data via GPIO."""
    raw_data = [int(x) for x in ZERO_SUB_RAW.split()]
    _print("Số lượng giá trị raw_data: {}".format(len(raw_data)))

    try:
        tx = OutputDevice(TX_PIN)
    except Exception as e:
        _print("Lỗi khởi tạo GPIO {}: {}".format(TX_PIN, e))
        return False

    _print("Phát tín hiệu trên GPIO {}...".format(TX_PIN))

    try:
        for duration in raw_data:
            if duration > 0:
                tx.on()
                time.sleep(duration / 1_000_000)
            else:
                tx.off()
                time.sleep(abs(duration) / 1_000_000)
    finally:
        tx.off()
        tx.close()

    _print("Replay tín hiệu hoàn tất!")
    return True


def get_updates(offset=None):
    params = {"timeout": 30}
    if offset is not None:
        params["offset"] = offset
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{TOKEN}/getUpdates",
            params=params, timeout=35
        )
        return resp.json().get("result", [])
    except Exception as e:
        _print("Polling error: {}".format(e))
        time.sleep(5)
        return []


def handle_message(message):
    chat_id = message["chat"]["id"]
    user_id = message["from"]["id"]
    text = message.get("text", "").strip()
    kb = get_main_keyboard()

    # Cập nhật tên hiển thị
    update_admin_name(user_id, message["from"])

    # /myid cho tất cả mọi người
    if text == "/myid":
        send_message(chat_id, "Chat ID của bạn: <code>{}</code>".format(user_id), kb)
        return

    # Chặn người không phải admin
    if user_id not in ADMIN_DB:
        send_message(chat_id, "Bạn không có quyền sử dụng bot này.\nID của bạn: <code>{}</code>".format(user_id))
        _print("Unauthorized: user_id={}".format(user_id))
        return

    # --- /start ---
    if text == "/start":
        help_text = (
            "<b>Door Lock Bot</b>\n\n"
            "/view - Chụp ảnh từ camera\n"
            "/open - Mở cửa\n"
            "/status - Kiểm tra cửa mở/đóng\n"
            "/trainopen - Train ảnh cửa mở (owner)\n"
            "/trainclose - Train ảnh cửa đóng (owner)\n"
            "/addadmin &lt;id&gt; - Thêm admin\n"
            "/removeadmin &lt;id&gt; - Xóa admin\n"
            "/log - Xem lịch sử hoạt động (owner)\n"
            "/listadmin - Xem danh sách admin\n"
            "/myid - Xem chat ID"
        )
        send_message(chat_id, help_text, kb)

    # --- /view ---
    elif text == "/view":
        send_message(chat_id, "Đang chụp ảnh...", kb)
        photo = capture_snapshot()
        if photo:
            write_log(user_id, "Xem camera")
            send_photo(chat_id, photo, caption="Snapshot - {}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        else:
            send_message(chat_id, "Không thể chụp ảnh từ camera!", kb)

    # --- /status ---
    elif text == "/status":
        send_message(chat_id, "Đang phân tích...", kb)
        photo = capture_snapshot()
        if photo:
            write_log(user_id, "Kiểm tra trạng thái cửa")
            status, msg = predict_door(photo)
            if status:
                send_photo(chat_id, photo, caption="Trạng thái: {}".format(status))
            send_message(chat_id, msg, kb)
        else:
            send_message(chat_id, "Không thể chụp ảnh từ camera!", kb)

    # --- /open ---
    elif text == "/open":
        global last_open_time
        elapsed = time.time() - last_open_time
        if elapsed < OPEN_COOLDOWN:
            remaining = int(OPEN_COOLDOWN - elapsed)
            send_message(chat_id, "Vui lòng chờ thêm {}s nữa.".format(remaining), kb)
            return

        last_open_time = time.time()  # set ngay để chặn spam
        t = threading.Thread(target=_open_door_thread, args=(chat_id, user_id, kb))
        t.daemon = True
        t.start()

    # --- /trainopen ---
    elif text == "/trainopen":
        if user_id != OWNER_ID:
            send_message(chat_id, "Chỉ owner mới có quyền train.", kb)
            return
        if not training_lock.acquire(blocking=False):
            send_message(chat_id, "Đang train rồi, vui lòng chờ...", kb)
            return
        send_message(chat_id, "Bắt đầu chụp 3 ảnh cửa MỞ (mỗi ảnh cách 17s)...", kb)
        write_log(user_id, "Train cửa mở")
        t = threading.Thread(target=_train_capture_thread, args=(chat_id, OPEN_DIR, "Cửa mở", kb))
        t.daemon = True
        t.start()

    # --- /trainclose ---
    elif text == "/trainclose":
        if user_id != OWNER_ID:
            send_message(chat_id, "Chỉ owner mới có quyền train.", kb)
            return
        if not training_lock.acquire(blocking=False):
            send_message(chat_id, "Đang train rồi, vui lòng chờ...", kb)
            return
        send_message(chat_id, "Bắt đầu chụp 3 ảnh cửa ĐÓNG (mỗi ảnh cách 17s)...", kb)
        write_log(user_id, "Train cửa đóng")
        t = threading.Thread(target=_train_capture_thread, args=(chat_id, CLOSE_DIR, "Cửa đóng", kb))
        t.daemon = True
        t.start()

    # --- /addadmin ---
    elif text.startswith("/addadmin"):
        if user_id != OWNER_ID:
            send_message(chat_id, "Chỉ owner mới có quyền thêm admin.", kb)
            return

        # Cách 1: /addadmin 123456
        parts = text.split()
        new_id = None
        if len(parts) == 2:
            try:
                new_id = int(parts[1])
            except ValueError:
                pass

        # Cách 2: reply tin nhắn của người khác
        if new_id is None and "reply_to_message" in message:
            new_id = message["reply_to_message"]["from"]["id"]

        if new_id is None:
            send_message(chat_id, "Cách dùng:\n/addadmin &lt;id&gt;\nHoặc reply tin nhắn của người cần thêm.", kb)
            return

        # Lấy tên nếu reply, không thì để trống
        new_name = ""
        if "reply_to_message" in message and message["reply_to_message"]["from"]["id"] == new_id:
            new_name = get_display_name(message["reply_to_message"]["from"])

        ADMIN_DB[new_id] = new_name
        save_admins()
        label = new_name if new_name else str(new_id)
        send_message(chat_id, "Đã thêm admin: {} (<code>{}</code>)".format(label, new_id), kb)
        _print("Admin added: {}".format(new_id))

    # --- /removeadmin ---
    elif text.startswith("/removeadmin"):
        if user_id != OWNER_ID:
            send_message(chat_id, "Chỉ owner mới có quyền xóa admin.", kb)
            return

        parts = text.split()
        if len(parts) != 2:
            send_message(chat_id, "Cách dùng: /removeadmin &lt;id&gt;", kb)
            return

        try:
            rm_id = int(parts[1])
        except ValueError:
            send_message(chat_id, "ID không hợp lệ.", kb)
            return

        if rm_id == OWNER_ID:
            send_message(chat_id, "Không thể xóa owner.", kb)
            return

        removed_name = ADMIN_DB.pop(rm_id, "")
        save_admins()
        label = removed_name if removed_name else str(rm_id)
        send_message(chat_id, "Đã xóa admin: {} (<code>{}</code>)".format(label, rm_id), kb)
        _print("Admin removed: {}".format(rm_id))

    # --- /listadmin ---
    elif text == "/listadmin":
        lines = []
        for aid in sorted(ADMIN_DB.keys()):
            name = ADMIN_DB[aid] or str(aid)
            role = " (owner)" if aid == OWNER_ID else ""
            lines.append("- {} (<code>{}</code>){}".format(name, aid, role))
        send_message(chat_id, "<b>Danh sách admin:</b>\n" + "\n".join(lines), kb)

    # --- /log ---
    elif text.startswith("/log"):
        if user_id != OWNER_ID:
            send_message(chat_id, "Chỉ owner mới có quyền xem log.", kb)
            return

        parts = text.split()
        n = 20
        if len(parts) == 2:
            try:
                n = int(parts[1])
            except ValueError:
                pass

        lines = read_log(n)
        if lines:
            send_message(chat_id, "<b>Log ({} dòng gần nhất):</b>\n<pre>{}</pre>".format(
                len(lines), "".join(lines)), kb)
        else:
            send_message(chat_id, "Chưa có log nào.", kb)

    else:
        send_message(chat_id, "Lệnh không hợp lệ. Dùng /start để xem hướng dẫn.", kb)


def main():
    set_bot_commands()
    _print("Bot started. Waiting for commands...")

    # Khởi động thread giám sát ban đêm
    night_thread = threading.Thread(target=_night_watch_thread)
    night_thread.daemon = True
    night_thread.start()

    offset = None

    while True:
        updates = get_updates(offset)
        for update in updates:
            offset = update["update_id"] + 1
            message = update.get("message")
            if not message:
                continue
            try:
                handle_message(message)
            except Exception as e:
                _print("Error handling message: {}".format(e))


if __name__ == "__main__":
    main()
