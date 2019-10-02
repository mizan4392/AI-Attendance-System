import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training-data.yml")
lables ={"DIU_ID",1}
with open("labels.pickle",'rb') as f:
    og_lables = pickle.load(f)
    lables = {v:k for k,v in og_lables.items()}
print(lables)
cap = cv2.VideoCapture(0)

while(True):
    # Capture Frame-by-Frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        gray_face_region = gray[y:y+h,x:x+w]
        color_face_region = frame[y:y + h, x:x + w]
        gray_image_item = "my-gray-image.png"
        color_image_item = "my-color-image.png"
        # make prediction on recognizer
        id_ , conf = recognizer.predict(gray_face_region)
        print(conf)
        if conf>=45 and conf<=105:
            print(id_)
            print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = lables[id_]
            color = (0,255,0)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        #cv2.imwrite(gray_image_item,gray_face_region)
        #cv2.imwrite(color_image_item, color_face_region)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    #display resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#release the capture
cap.release()
cv2.destroyAllWindows()
