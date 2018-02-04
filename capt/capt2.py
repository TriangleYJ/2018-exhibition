import time
import cv2
import mss
import numpy


with mss.mss() as sct:
    monitor = {'top': 40, 'left': 0, 'width': 100, 'height': 100}
    img_matrix = []
    last_time = time.time()
    while True:

        # Get raw pizels from screen and save to numpy array
        img = numpy.array(sct.grab(monitor))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save img data as matrix
        img_matrix.append(img)

        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        # Display Image
        cv2.imshow('Normal', img)

        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
