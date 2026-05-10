"""Reusable MediaPipe hand-tracking helper for MathCam."""

import cv2 as cv
import mediapipe as mp


class handDetector:
    """Small wrapper around MediaPipe Hands used by the drawing app."""

    def __init__(self, mode=False, maxHands=2, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None
        self.handLabel = "Right"

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if not self.results or not self.results.multi_hand_landmarks:
            return self.lmList

        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList

        if self.results.multi_handedness and handNo < len(self.results.multi_handedness):
            self.handLabel = self.results.multi_handedness[handNo].classification[0].label

        myHand = self.results.multi_hand_landmarks[handNo]
        h, w, _ = img.shape
        for landmark_id, landmark in enumerate(myHand.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            self.lmList.append([landmark_id, cx, cy])
            if draw:
                cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return self.lmList

    def fingersUp(self):
        """Return a five-item list indicating whether each finger is raised."""
        if len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb direction depends on handedness in a mirrored camera image.
        if self.handLabel == "Right":
            fingers.append(int(self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]))
        else:
            fingers.append(int(self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]))

        for finger_id in range(1, 5):
            fingers.append(
                int(self.lmList[self.tipIds[finger_id]][2] < self.lmList[self.tipIds[finger_id] - 2][2])
            )
        return fingers


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Camera not available.")
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if lmList:
            print(lmList[4])

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
