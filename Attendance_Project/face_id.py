import cv2
import face_recognition
import os

path = "images"
images = []
classNames = []

# load images
for img_name in os.listdir(path):
    img = cv2.imread(f"{path}/{img_name}")
    images.append(img)
    classNames.append(os.path.splitext(img_name)[0])

# encode faces
encodings = []
for img in images:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)[0]
    encodings.append(enc)

# webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    small = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb_small)
    face_encs = face_recognition.face_encodings(rgb_small, faces)

    for encodeFace, faceLoc in zip(face_encs, faces):
        matches = face_recognition.compare_faces(encodings, encodeFace)
        name = "Unknown"

        if True in matches:
            matchIndex = matches.index(True)
            name = classNames[matchIndex]

        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, name, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("AI Identity Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()