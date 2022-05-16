import cv2
import mediapipe as mp
import time

class HandTracker:
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        # Setting up media pipe hand tracker
        self.mpHands = mp.solutions.hands
        # by default static_image_mode is set to false so that detection occurs only when tracking
        # fall below a certain threshhold
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        # To draw lines between 21 points
        self.mpDraw = mp.solutions.drawing_utils
        self.hand_num = 0
        self.results = None
    
    def find_hands(self, img, draw=True):
        '''
        This method takes in an image finds the hand landmark
        in the image and stores it in self.results, draws lines 
        between the detected landmark points if necessary.
        '''
        # convert img to RGB and feed it to media-pipe hand processing to obtain the results
        self.results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            # gives the number of hands detected
            self.hand_num = len(self.results.multi_hand_landmarks)
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, hand, self.mpHands.HAND_CONNECTIONS)
        if self.results.multi_handedness:
            print(f'Detected {self.hand_num} hands.')
        return img

def test():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandTracker()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Images", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    test()
