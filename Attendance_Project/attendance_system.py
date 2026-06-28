import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os

def mark_attendance(name):
    now = datetime.now()

    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists("attendance.csv"):
        with open("attendance.csv","w") as f:
            f.write("Name,Date,Time\n")

    df = pd.read_csv("attendance.csv")

    if name not in df["Name"].values:
        with open("attendance.csv","a") as f:
            f.write(f"{name},{date},{time}\n")
# Load images
images = []
classNames = []

path = "images"

myList = os.listdir(path)
print(myList)
for file in myList:
    img = cv2.imread(f"{path}/{file}")
    images.append(img)
    classNames.append(os.path.splitext(file)[0])

print(classNames)
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        import numpy as np
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:

            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc

            y1 *= 4
            x2 *= 4
            y2 *= 4
            x1 *= 4

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)

            cv2.putText(img,name,(x1+6,y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,(255,255,255),2)

            mark_attendance(name)

    cv2.imshow("Attendance System",img)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()