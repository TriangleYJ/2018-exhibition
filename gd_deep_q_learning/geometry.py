import time
import cv2
import mss
import numpy as np
import pyautogui
import openpyxl
from collections import deque


def current_milli_time(): # 현재의 시간을 소수 셋째짜리까지 구하는 함수입니다.
    return int(round(time.time() * 1000)) / 1000

def checkvalue(array): # 픽셀의
    arraya = []
    for i in range(480):
        pix = array[i][0]
        arraya.append((pix[0] - 255) ** 2 + (pix[1] - 255) ** 2 + (pix[2] - 0) ** 2)
    return min(arraya)

class geometry:
    pyautogui.PAUSE = 0.00001
    now = 0
    init = current_milli_time()
    active = 0
    a = 1 #엑셀 시작 지점

    D = deque()


    def frame_step(self, input_actions):

        with mss.mss() as sct:
            if input_actions[1] == 1:
                pyautogui.click(216,445)
            monitor = {'top': 31, 'left': 10, 'width': 720, 'height': 480}

            reward = 0
            terminal = False
            self.now = current_milli_time() - self.init

            # Get raw pizels from screen and save to numpy arrayqq
            img = np.array(sct.grab(monitor))
            crop_xt2 =cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
            crop_xt1 = img[0:480, 245:246]

            """cv2.imshow('window', crop_xt2)
            if(cv2.waitKey(25)&0xFF==ord('q')):
                cv2.destroyAllWindows()"""
            cnow = checkvalue(crop_xt1)
            self.D.append(cnow)
            last = 0
            if len(self.D) > 6:
                last = self.D.popleft()

            if (checkvalue(crop_xt1) != 0) and self.now > 2000/1000 and last == cnow: # dead

                terminal = True
                if self.active > 0:
                    document = openpyxl.load_workbook('Test.xlsx')

                    ws1 = document.active
                    ws1.cell(row=self.a, column=self.active).value = self.now
                    document.save('Test.xlsx')
                    self.active -= 1
                    if self.active == 0: self.a += 1
                    print("==============recorded!================")
                reward = (-1) * self.now
                self.init = current_milli_time()

            else:
                reward = self.now


            return crop_xt2, reward, terminal