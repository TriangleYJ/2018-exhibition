import numpy as np
from PIL import ImageGrab
import cv2
import time
import re
srt = (100, 100)
size = (900, 810)

def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(srt[0], srt[1], srt[0] + size[0], srt[1] + size
                                                    [1])))
        cv2.imshow('window',cv2.cvtColor(cv2.resize(printscreen, (900, 810)), cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

#screen_record()


screen_record()