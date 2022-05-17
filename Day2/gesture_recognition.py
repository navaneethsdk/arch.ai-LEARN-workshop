import cv2
import mediapipe as mp
import time
from hand_tracker import HandTracker
import pickle

p_time = 0
c_time = 0
cap = cv2.VideoCapture(0)
detector = HandTracker()
model = pickle.load(open('Models/model.sav', 'rb'))
finish = False
zero_padding = [0 for i in range(42)]
hand_no = 0
while not finish:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.get_raw_data()
    if len(lm_list)==42:
        lm_list = lm_list+zero_padding
    if len(lm_list) != 0:
        try:
            cv2.putText(img, model.predict(
                [lm_list])[0], (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        except ValueError:
            print("Unexpected features")
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Images", img)
    if cv2.waitKey(1) == 27:
        finish = True
