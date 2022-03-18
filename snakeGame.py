import math
import random

import cvzone
import cv2
import numpy as np

import mediapipe as mp
import points as points
from cvzone.HandTrackingModule import HandDetector

# configurar camara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# detector de manos y confidencia para solo usar una mano
detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # todos los puntos de la snake
        self.lengths = []  # distancia entre cada punto
        self.currentLength = 0  # longitud total del snake
        self.allowedLengths = 150  # longitud total permitida
        self.previousHead = 0, 0  # punto previo head del snake
        self.imgFood = cv2.imread(
            pathFood,
            cv2.IMREAD_UNCHANGED
        )
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()
        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.gameOver:
            cvzone.putTextRect(imgMain, "GAME OVER", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Tuu puntaje es: {self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Reduccion de longitud
            if self.currentLength > self.allowedLengths:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLengths:
                        break

            # check si snake comio la fruta
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLengths += 50
                self.score += 1
                print(self.score)
                # print("#Yae me lo comi")

            # Dibujar Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(
                            imgMain,
                            self.points[i - 1],
                            self.points[i],
                            (0, 0, 255),
                            20
                        )
                cv2.circle(
                    imgMain,
                    self.points[-1],
                    20,
                    (200, 0, 200),
                    cv2.FILLED
                )

            # Dibujar Fruta
            imgMain = cvzone.overlayPNG(
                imgMain,
                self.imgFood,
                ((rx - self.wFood) // 2, (ry - self.hFood) // 2)
            )
            cvzone.putTextRect(imgMain, f'Tuu puntaje es: {self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)

            # check si hay colision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(
                imgMain,
                [pts],
                False,
                (0, 200, 0),
                3
            )
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            # print(minDist)

            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True

                self.points = []
                self.lengths = []
                self.currentLength = 0
                self.allowedLengths = 150
                self.previousHead = 0, 0
                self.randomFoodLocation()
                self.score = 0

        return imgMain


# crear Snake en game
game = SnakeGameClass('fruta2.png')
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(
        img,
        flipType=False
    )

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]  # posIndex X_Y
        img = game.update(img, pointIndex)

    k = cv2.waitKey(1)
    if k == 27:
        break

    k1 = cv2.waitKey(1)
    if k1 == 32:
        game.gameOver=False

    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver=True

    cv2.imshow("Image", img)
    cv2.waitKey(1)
