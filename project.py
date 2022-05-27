import numpy as np
import cv2
import face_recognition
import os
import datetime


def findEncoding(images):
    encodeList = []
    for img in images:
        imgTemp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(imgTemp)[0]
        encodeList.append(encode)
    return encodeList


def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dateStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateStr}')


path = 'ImagesProject'
images = []
className = []
myList = os.listdir(path)

print(myList)

for cls in myList:
    myImg = os.listdir(path + '/' + cls)
    for img in myImg:
        curImg = cv2.imread(f'{path}/{cls}/{img}')
        images.append(curImg)
        className.append(os.path.splitext(img)[0])

    encodeListKnown = findEncoding(images)

# print(encodeListKnown)

# print(className)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    igS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    igS = cv2.cvtColor(igS, cv2.COLOR_BGR2RGB)

    faceCurFrames = face_recognition.face_locations(igS)
    encodeCurFrame = face_recognition.face_encodings(igS, faceCurFrames)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrames):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

        # print(faceDist)

        matchIdx = np.argmin(faceDist)

        if matches[matchIdx]:
            name = className[matchIdx].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
