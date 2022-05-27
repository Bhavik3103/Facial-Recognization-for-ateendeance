import os
import time
import cv2
import joblib
import numpy as np
import sqlite3 as sql
import pyttsx3
from PIL import Image
from tkinter import *
from sklearn.ensemble import RandomForestClassifier
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder


def text_to_speech(user_text):
    engine = pyttsx3.init()
    engine.say(user_text)
    engine.runAndWait()


# ======================== To Find the Facees ==============================================

# Loading facenet model for detecting face
embedder = FaceNet()
# embedding_model = load_model('facenet_keras.h5')
print('Embedding Model Loaded')

# making a mtcnn instance for detecting faces
detector = MTCNN()

# --> FaceNet was deloped by the Google

# Here the image size is (160 x 160) as the facenet model expects this size of the image


def find_face(img,img_size=(160,160)):
    img = cv2.imread(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = np.asarray(img) # converting our image obj to numpy array
    faces = detector.detect_faces(img)
    if faces:
        x,y,w,h = faces[0]['box']
        x,y=abs(x),abs(y)
        face = img[y:y+h,x:x+w]
        face = Image.fromarray(face) # converting it to image object to resize it
        face = face.resize(img_size) # resizing it
        face = np.asarray(face)      # converting it back to array
        return face
    return None

# ======================== To find Face Embeddings ==========================


def embed(face):
    face = face.astype('float32')
    fm, fs = face.mean(), face.std()
    face = (face-fm)/fs  # standardizing the data
    face = np.expand_dims(face, axis=0)  # flattening it
    # embedding model converts our160*160*3 vector to 128 features
    # embs = embedding_model.predict(face)
    embs = embedder.embeddings(face)
    return embs[0]

# ======================== To Load Dataset to train and Validate ==========================


def load_dataset(path):
    X = []
    y = []
    for people in os.listdir(path):
        for people_images in os.listdir(path+people):
            face = find_face(path+people+'/'+people_images)
            if face is None:
                continue
            emb = embed(face)
            X.append(emb)
            y.append(people)
        print('Loaded {} images of {}'.format(
            len(os.listdir(path+'/'+people)), people))
    return np.asarray(X), np.asarray(y)


def addNewStudent():
    if USERNAME.get() == '' or NAME.get() == '' or PASSWORD.get() == '':
        lbl_err = Label(error, text="Please enter all the details !", fg="red")
        lbl_err.pack()
    else:
        print("Nothing yet ...")


# ======================== Root window ==========================
root = Tk()
root.title("Add New Student ")
root.resizable(width=FALSE, height=FALSE)
root.geometry("%dx%d+%d+%d" % (500, 600, 500, 100))
root.configure(background="#b6effa")

# ========================== Frames of window  =========================
heading = Frame(root, bd=10)
heading.pack(side=TOP, fill=X)
error = Frame(root)
error.pack(side=TOP, fill=X)
details = Frame(root, height=335, width=400)
details.pack(side=TOP, ipady=10, ipadx=10, pady=100)

# ========================= Labels =================================
title = Label(heading, text="Add Details", bg="white",
              fg="#0a29c4", font=('Helvetica', 48))
title.pack(side=TOP, fill=BOTH, expand=1)

name = Label(details, text="Name ", font=('Helvetica', 14), pady=10)
name.grid(row=0, sticky="e")
roll = Label(details, text="Roll No. ", font=('Helvetica', 14), pady=10)
roll.grid(row=1, sticky="e")
user = Label(details, text="Username ", font=('Helvetica', 14), pady=10)
user.grid(row=2, sticky="e")
password = Label(details, text="Password ", font=('Helvetica', 14), pady=10)
password.grid(row=3, sticky="e")

# ============================== Variables ===================================
NAME = StringVar()
ROLL = StringVar()
USERNAME = StringVar()
PASSWORD = StringVar()
global con, c


# =============================  Values entered from user ====================================
username = Entry(details, text=USERNAME, font=14, relief=RIDGE)
username.grid(row=2, column=1)
Pass = Entry(details, text=PASSWORD, show="*", font=14, relief=RIDGE)
Pass.grid(row=3, column=1)
Name = Entry(details, text=NAME, font=(14), relief=RIDGE)
Name.grid(row=0, column=1)
Roll = Entry(details, text=ROLL, font=14, relief=RIDGE)
Roll.grid(row=1, column=1)

# =============================== On Clicking Submit ========================================

def startCamera(nameStr):
    cap = cv2.VideoCapture(0)
    i = 0
    print()
    for i in range(5):
        print(f'Capturing starts in {5-i} seconds...')
        time.sleep(1)
    print('Taking photos...')
    while i <= 200:
        ret, frame = cap.read()
        cv2.imshow('taking your pictures', frame)
        if i % 5 == 0 and i <= 150 and i != 0:
            cv2.imwrite('faces/train/'+nameStr+'/'+str(i)+'.png', frame)
        elif i % 5 == 0 and i > 150:
            cv2.imwrite('faces/val/'+nameStr+'/'+str(i)+'.png', frame)
        i += 1
    cv2.destroyAllWindows()
    cap.release()
    print('Successfully taken your photos...')


def addDetails():
    if USERNAME.get() == '' or NAME.get() == '' or PASSWORD.get() == '':
        lbl_err = Label(error, text="Please enter all the details !", fg="red")
        lbl_err.pack()
    else:
        global con, c
        con = sql.connect('LogInDataBase.db')
        c = con.cursor()
        params = (USERNAME.get(), PASSWORD.get())
        c.execute(f"""INSERT INTO `Student` (username , password) 
            VALUES (? , ?)
            """, params)
        con.commit()
        c.close()
        con.close()
        nameStr = NAME.get() + "-" + ROLL.get()
        if nameStr in os.listdir('faces/train'):
            lbl_err = Label(error, text="User already exist in faces !", fg="red")
            lbl_err.pack()
            print('User already exist in faces')
        else:
            # root.quit()
            os.makedirs('faces/train/'+nameStr)
            os.makedirs('faces/val/'+nameStr)
            # Starting the Camera
            startCamera(nameStr)
            # ============================= Loading Dataset ====================================
            print('Loading train data...')
            X_train, y_train = load_dataset('faces/train/')

            print()

            print('Loading test data...')
            X_test, y_test = load_dataset('faces/val/')

            # l2 normalizing the data
            l2_normalizer = Normalizer('l2')

            X_train = l2_normalizer.transform(X_train)
            X_test = l2_normalizer.transform(X_test)

            # label encoding the y data
            label_enc = LabelEncoder()
            y_train = label_enc.fit_transform(y_train)
            y_test = label_enc.transform(y_test)

            ############ Training SVC (Support Vector Classifier) for predicting faces ########
            rfc = RandomForestClassifier()
            rfc.fit(X_train, y_train)

            joblib.dump(rfc, 'models/face_prediction_model.sav')
            print()

            print('Random Forest Model saved successfully!!')

            text_to_speech("New user added SUCCESSfully!!!")
            root.quit()


# =============================== Buttons =================================================
btn_back = Button(details, text="SUBMIT", width=45, command=addDetails)
btn_back.grid(row=5, columnspan=2, pady=25, padx=10)


root.mainloop()
