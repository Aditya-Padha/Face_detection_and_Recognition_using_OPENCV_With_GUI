from tkinter.ttk import *
import cv2
import os
import numpy as np
from PIL import Image, ImageTk                            # Mini-Project by: Aditya Padha
import tkinter

def Trainer():
    # path for face image database
    path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def getImagesandLabels(path):
        imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
        facesamples = []                             # function to get images and label data
        ids = []
        for imagePath in imagepaths:
            PIL_img = Image.open(imagePath).convert('L')  # grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                facesamples.append((img_numpy[y:y + h, x:x + w]))
                ids.append(id)
        return facesamples, ids
    print("\n [INFO] Training faces. it will take a few seconds Wait...")
    faces, ids = getImagesandLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')   # Save the model into trainer/trainer.yml
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

def TakeImages():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    name = e1.get()
    face_id = int(e2.get())
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    count = 0              # Initialize individual sampling face count
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 100:  # Take 100 face sample and stop video
            break
    l=[(" "+name)]           # Do a bit of cleanup
    file1 = open("data.txt","a+")
    file1.writelines(l)
    file1.close()
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    cascadepath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadepath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0  # indicate id counter
    names=[]
    my_file = open("data.txt","r")
    data = my_file.read()
    names = data.split(" ")
    print(names)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)
    cam.set(4, 480)
    # define min window size to be recognized as face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=10,minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
                id = names[id]
            else:
                id = "unknown"
            cv2.putText(img,str(id),(x, y - 5),font,1,(255, 255, 255),2)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

r=tkinter.Tk()
r.title("Face Detection and Recognition")
r.geometry("360x180")
r.eval("tk::PlaceWindow . center")
img=Image.open(r"Background.png")
img=img.resize((360,180))                 # Background
bg_img=ImageTk.PhotoImage(img)
bg=Label(r,image=bg_img,justify="center")
bg.place(x=0,y=0)
l1 = Label(r, text="Enter Name:")
l2 = Label(r, text="Enter the ID:")
l1.grid(row=0, column=0, pady=2)
l2.grid(row=1, column=0, pady=2)
e1 = Entry(r)
e2 = Entry(r)
e1.grid(row=0, column=1, pady=2)
e2.grid(row=1, column=1, pady=2)
l3 = Label(r)
l3.grid(row=2,column=0)
bt1 = tkinter.Button(r, text="Take Images", command=TakeImages)
bt1.grid(row=7, column=0)
bt2 = tkinter.Button(r, text="Train The Model", command=Trainer)
bt2.grid(row=7, column=1)
bt3 = tkinter.Button(r, text="Face Recognition", command=face_recognition)
bt3.grid(row=7, column=2)
r.mainloop()
