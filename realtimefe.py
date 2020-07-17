# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:46:01 2020

@author: bodda
"""

from keras.models import load_model
from keras.preprocessing import  image
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier(r'E:\faceexpression\haarcascade_frontalface_alt.xml')
model=load_model(r'E:\face rec\fe.h5')

classes=['mani','pavani','riki']
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(frame,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=frame[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(74,74),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            
            preds=model.predict(roi)[0]
            print(preds)
            
            label=classes[preds.argmax()]
            print(label)
            label_pos=(x,y)
            cv2.putText(frame,label,label_pos,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
             cv2.putText(frame,"no face found",label_pos,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow("face rec",frame)
   
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
