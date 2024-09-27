#import Libraries
import cv2 
import numpy as np
from tensorflow.keras.models import model_from_json  # type: ignore 

print('TEST Import Library Success')

# Dictionary Data
# ENG_VERSION
# dict_data = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# IND_VERSION
dict_data = {0:'Marah', 1:'Jijik', 2:'Takut', 3:'Senang', 4:'Sedih', 5:'Terkejut', 6:'Netral'}

## Load JSON
json_file = (open('fileData/model.json', 'r'))
load_json = json_file.read()
json_file.close()

model = model_from_json(load_json)

## Load Weights
model.load_weights('fileData/model.weights.h5')
print("TEST Load Model Success")


## Test with Webcam
Vidcam = cv2.VideoCapture(0) 


## Test with Video
# test_video = cv2.VideoCapture('D:\\Code\\emotion\\Video\\vid.mp4')

cam = Vidcam
while True :
    ret, frame = cam.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    detect_face = cv2.CascadeClassifier('haarcascade_face_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## detect face on camera
    faces = detect_face.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in faces :
        cv2.rectangle(frame, (x,y), (x+w, y+h+10), (255,0,0), 4)
        gray_frame = gray[y:y+h, x:x+w]
        cropped_image = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48,48)), -1), 0)

        ## Predict
        predict = model.predict(cropped_image)
        max_index = int(np.argmax(predict))
        cv2.putText(frame, dict_data[max_index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Pendeteksi Emosi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

