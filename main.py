import numpy as np
import cv2
import face_recognition

imgMusk = face_recognition.load_image_file('PSC-Project/musk.jpg')
imgMusk = cv2.cvtColor(imgMusk,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('PSC-Project/musk_test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# face_locations gives the tupe (Top , Right , Bottom , Left)
faceLoc = face_recognition.face_locations(imgMusk)[0]
encodeMusk = face_recognition.face_encodings(imgMusk)[0]

#rectangle( image_src , tuple of image cordinate values (left , top , right , bottom) , color , line_width)
cv2.rectangle(imgMusk , (faceLoc[3] , faceLoc[0] , faceLoc[1] , faceLoc[2]) , (245 , 66 , 102) , 2)
# print(encodeMusk)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest , (faceLocTest[3] , faceLocTest[0] , faceLocTest[1] , faceLocTest[2]) , (245 , 66 , 102) , 2)

# Compares the encoding of two faces found in two images 
# Here the first argument is list
result = face_recognition.compare_faces([encodeMusk] , encodeTest)
faceDist = face_recognition.face_distance([encodeMusk] , encodeTest)
# putText = (src , Text , origin , font_style , scale color , line_width )
cv2.putText(imgTest , f'{result}  {round(faceDist[0] , 2)}' , (50 , 50) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0 , 0 , 255) , 2)
print(faceDist)

cv2.imshow('Elon Musk' , imgMusk)
cv2.imshow('Elon Test' , imgTest)
cv2.waitKey(0)