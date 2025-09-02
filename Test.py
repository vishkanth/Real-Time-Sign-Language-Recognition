import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Hand detector and classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("D:\\Indian Sign Detection-master(FINAL)\\IndianSignDetection-master\\Model\\keras_model.h5",
                        "D:\\Indian Sign Detection-master(FINAL)\\IndianSignDetection-master\\Model\\labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["Hello", "I'm", "C", "A", "R", "O", "L"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # White canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure cropping does not go out of bounds
        height, width, _ = img.shape
        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, width)
        y2 = min(y + h + offset, height)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("Warning: Empty crop. Skipping frame.")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Drawing on output
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Display
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()