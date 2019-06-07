# Face Recognition

import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
ns=cv2.CascadeClassifier('Nariz.xml')
mo=cv2.CascadeClassifier('Mouth.xml')
sm=cv2.CascadeClassifier('smile.xml')
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 21)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (120, 25, 240), 2)
        nose=ns.detectMultiScale(roi_gray,1.1,3)
        for(nx,ny,nw,nh) in nose:
           cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(146,129,86),2)

        smile=sm.detectMultiScale(roi_gray,1.6,15)
        for(sx,sy,sw,sh) in smile:
           cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),2)
    return frame

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output = detect(gray, frame)
    cv2.imshow('Video', output)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
video_capture.release()
cv2.destroyAllWindows()
