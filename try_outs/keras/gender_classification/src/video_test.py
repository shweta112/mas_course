import cv2
from keras.models import load_model
import numpy as np
from statistics import mode

detection_model_path = '../trained_models/haarcascade_frontalface_default.xml'
classification_model_path = '../trained_models/model_CNN.hdf5'
frame_window = 10
gender_labels = {0:'Female',1:'Male'}

face_detection = cv2.CascadeClassifier(detection_model_path)
gender_classifier = load_model(classification_model_path)
gender_window = []
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.startWindowThread()
cv2.namedWindow('window_frame')
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        try:
            face = cv2.resize(face,(64,64))
        except:
            continue
        face = np.expand_dims(face,0)
        face = np.expand_dims(face,-1)
        #face = preprocess_input(face)
        #face = face - 143
        gender_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_arg]
        gender_window.append(gender)

        if len(gender_window) >= frame_window:
            gender_window.pop(0)
        try:
            gender_mode = mode(gender_window)
        except:
            continue
        cv2.putText(gray,gender_mode,(x,y-30), font, .7,(255,0,0),1,cv2.LINE_AA)
    try:
        cv2.imshow('window_frame', gray)
    except:
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

