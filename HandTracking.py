import cv2
import mediapipe as mp
from time import time


class HandTracking:
    def __init__(self):
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()
        self.draw = mp.solutions.drawing_utils

    def detectionMod(self, image):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imageRGB)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.draw.draw_landmarks(image, hand, self.mphands.HAND_CONNECTIONS)
                for handId, location in enumerate(hand.landmark):
                    x, y, _ = image.shape
                    cx, cy = int(location.x * y), int(location.y * x)
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    cv2.putText(image, str(handId + 1), (cx + 5, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        return image

    def videoFeed(self):
        cam = cv2.VideoCapture(0)
        ptime = time()
        try:
            while True:
                successfulFrame, image = cam.read()
                image = self.detectionMod(cv2.flip(image, 1))
                ctime = time()
                fps = 1 / (ctime - ptime)
                ptime = ctime
                cv2.putText(image, str(int(fps)), (10, 75), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
                cv2.imshow("VideoFeed", image)
                if cv2.waitKey(1) != -1:
                    break
        finally:
            cam.release()


if __name__ == "__main__":
    HandTracking().videoFeed()
