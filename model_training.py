import os
from imutils import paths
import face_recognition
import pickle
import cv2

print("[INFO] start processing faces...")
imagePaths = list(paths.list_images("dataset"))
knownEncodings = []
known_face_names = []
data_file = "encodings.pickle"
if os.path.exists(data_file):
    print("[INFO] loading encodings...")
    with open(data_file, "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
knownNames = []
new_face = False
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    if name in known_face_names:
        continue
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        new_face = True
if new_face:
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    with open("encodings.pickle", "wb") as f:
        f.write(pickle.dumps(data))

    print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")
else:
    print("no new face need to train")