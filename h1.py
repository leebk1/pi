# ---------------------------------- make2explore.com----------------------------------------------------------#
# Tutorial          - MediaPipe Machine Learning Solutions on Raspberry Pi - Example 3 : Hand Pose 2
# Version - 1.0
# Last Modified     - 24/03/2022 15:00:00 @admin
# Software          - Python, Thonny IDE, Standard Python Libraries, MediaPipe Python Package
# Hardware          - Raspberry Pi 4 model B, Logitech c270 webcam
# Source Repo       - https://github.com/make2explore/MediaPipe-On-RaspberryPi
# Credits           - MediaPipe Framework by Google : https://mediapipe.dev/
# -------------------------------------------------------------------------------------------------------------#
import sys

import pigpio
import time
import cv2
import mediapipe as mp
import pigpio
import time


def send_trigger():
    FILE_NAME = "zero.sub"  # Tên file chứa dữ liệu capture từ Flipper Zero

    # --- 1. Đọc file và trích xuất raw_data ---
    with open(FILE_NAME, "r") as f:
        lines = f.readlines()

    raw_data_str = None
    frequency = None
    preset = None
    protocol = None

    for line in lines:
        line = line.strip()
        if line.startswith("RAW_Data:"):
            # Lấy phần dữ liệu sau "RAW_Data:"
            raw_data_str = line[len("RAW_Data:"):].strip()
        elif line.startswith("Frequency:"):
            frequency = line[len("Frequency:"):].strip()
        elif line.startswith("Preset:"):
            preset = line[len("Preset:"):].strip()
        elif line.startswith("Protocol:"):
            protocol = line[len("Protocol:"):].strip()

    if raw_data_str is None:
        print("Không tìm thấy dòng RAW_Data trong file!")
        exit(1)

    print("Load file {}:".format(FILE_NAME))
    print("  Frequency =", frequency)
    print("  Preset    =", preset)
    print("  Protocol  =", protocol)

    # Tách chuỗi thành các giá trị và chuyển thành số nguyên.
    raw_data_values = raw_data_str.split()
    try:
        raw_data = [int(x) for x in raw_data_values]
    except Exception as e:
        print("Lỗi chuyển đổi dữ liệu:", e)
        exit(1)

    print("Số lượng giá trị raw_data:", len(raw_data))

    # --- 2. Cấu hình pigpio và chân TX ---
    TX_PIN = 27  # Chân GPIO dùng để phát tín hiệu
    pi = pigpio.pi()
    if not pi.connected:
        print("Không kết nối với pigpio daemon. Hãy chạy 'sudo pigpiod'.")
        exit(1)

    pi.set_mode(TX_PIN, pigpio.OUTPUT)

    # --- 3. Tạo waveform từ raw_data ---
    #
    # Ở định dạng file capture bởi Flipper Zero, mỗi giá trị:
    #   - Nếu dương: bật TX_PIN (HIGH) trong khoảng thời gian tương ứng (micro giây).
    #   - Nếu âm: tắt TX_PIN (LOW) trong khoảng thời gian tương ứng (lấy giá trị tuyệt đối).
    #
    pulses = []
    for duration in raw_data:
        if duration > 0:
            pulses.append(pigpio.pulse(1 << TX_PIN, 0, duration))
        else:
            pulses.append(pigpio.pulse(0, 1 << TX_PIN, abs(duration)))

    print("Tạo waveform từ raw_data và phát tín hiệu trên GPIO {}...".format(TX_PIN))

    pi.wave_add_new()
    pi.wave_add_generic(pulses)
    wave_id = pi.wave_create()

    if wave_id < 0:
        print("Lỗi tạo waveform!")
        pi.stop()
        exit(1)

    # --- 4. Phát lại waveform ---
    pi.wave_send_once(wave_id)
    while pi.wave_tx_busy():
        time.sleep(0.01)

    # Đặt TX_PIN về mức LOW và dọn dẹp waveform, sau đó đóng kết nối với pigpio
    pi.wave_delete(wave_id)
    pi.write(TX_PIN, 0)
    pi.stop()
    print("Replay tín hiệu hoàn tất!")



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def is_five_fingers_open(hand_landmarks):
    # Lấy các điểm đặc trưng của bàn tay
    landmarks = hand_landmarks.landmark

    # Kiểm tra các ngón tay
    fingers_open = {
        'thumb': True,
        'index': False,
        'middle': False,
        'ring': False,
        'pinky': False
    }

    # Điều kiện cho từng ngón tay
    # Ngón cái: So sánh vị trí x của đầu ngón và đốt ngón
    fingers_open['thumb'] = landmarks[4].y < landmarks[1].y

    # Các ngón khác: So sánh vị trí y của đầu ngón và đốt giữa
    fingers_open['index'] = landmarks[8].y < landmarks[5].y
    fingers_open['middle'] = landmarks[12].y < landmarks[9].y
    fingers_open['ring'] = landmarks[16].y < landmarks[13].y
    fingers_open['pinky'] = landmarks[20].y < landmarks[17].y



    # print(fingers_open)
    # Đếm số ngón xòe
    open_count = sum(fingers_open.values())
    # return True
    return open_count == 5
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands. Hands (
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5 ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        if is_five_fingers_open(hand_landmarks):
            # Lấy tọa độ bounding box
            h, w, c = image.shape
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)

            # Vẽ khung và text
            cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.putText(image, '5 FINGERS DETECTED', (x_min - 20, y_min - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            send_trigger()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
