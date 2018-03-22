import time
import cv2
import mss
import numpy as np
from collections import deque
import pyautogui

class geometry:
    pyautogui.PAUSE = 0.03333
    REPLAY_DELAYCOUNT = 1
    D = deque()
    mass_start = 0.2
    mass_increase = 0.0001


    def frame_step(self, input_actions):
        with mss.mss() as sct:
            if input_actions[1] == 1:
                pyautogui.click(100,100)
            monitor = {'top': 35, 'left': 0, 'width': 720, 'height': 480}

            terminal = False
            reward = 0

            self.mass_start += self.mass_increase

            # Get raw pizels from screen and save to numpy array
            img = np.array(sct.grab(monitor))
            x_t1 = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
            crop_xt1 = x_t1[70:80, 70:80] # custom position

            self.D.append(crop_xt1)

            if len(self.D) > self.REPLAY_DELAYCOUNT:
                ors = self.D.popleft()
                if np.all(ors == crop_xt1): # dead
                    terminal = True
                    self.mass_start = 0.2
                    reward = -0.5-self.mass_start
                else:
                    reward = self.mass_start

            return x_t1, reward, terminal
