import cv2
import os

# Nhập tên user và tạo thư mục dataset nếu chưa có
user_name = input("Nhập tên user: ").strip()
dataset_path = f"./dataset/{user_name}"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
    print(f"Đã tạo thư mục: {dataset_path}")

# Khởi tạo camera (sử dụng USB camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera!")
    exit(1)

# Khởi tạo Haar Cascade để phát hiện khuôn mặt
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Biến đếm ảnh đã lưu
count = 1

print("Nhấn Space để lưu khuôn mặt, nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame từ camera!")
        break

    # Lật frame cho hiệu ứng mirror (tùy chỉnh nếu cần)
    # frame = cv2.flip(frame, 1)
    #
    # # Chuyển sang grayscale để phát hiện khuôn mặt
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # Phát hiện khuôn mặt
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # # Vẽ khung hình quanh khuôn mặt phát hiện được
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #
    # # Thông báo hướng dẫn
    # cv2.putText(frame, "Press SPACE to capture face, ESC to exit",
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key để thoát
        print("Thoát...")
        break
    elif key == 32:  # SPACE key để capture
        # if len(faces) == 0:
        #     print("Không phát hiện khuôn mặt nên không lưu ảnh!")
        #     continue
        # # Lấy khuôn mặt đầu tiên (nếu có nhiều)
        # (x, y, w, h) = faces[0]
        # face_img = gray[y:y + h, x:x + w]
        save_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Đã lưu khuôn mặt vào: {save_path}")
        count += 1

cap.release()
cv2.destroyAllWindows()
