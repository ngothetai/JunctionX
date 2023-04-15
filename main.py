import cv2
import numpy as np
import threading

class VideoCaptureThread(threading.Thread):
    def __init__(self, src, width, height):
        threading.Thread.__init__(self)
        self.video = cv2.VideoCapture(src)
        self.width = width
        self.height = height
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def run(self):
        ret, frame = self.video.read()
        if not ret:
            self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            self.frame = cv2.resize(frame, (self.width, self.height))   

class MultiVideoCapture:
    def __init__(self, width, height, *args):
        self.n = len(args)
        self.width = width
        self.height = height
        self.grid_width = int(self.width/2)
        self.grid_height = int(self.height/2)

        self.background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.background = cv2.resize(self.background, (self.width, self.height))
        
        self.threads = []
        for cap in args:
            self.threads.append(VideoCaptureThread(cap, width=self.grid_width, height=self.grid_height))

    def start(self):
        for thread in self.threads:
            thread.start()

    def display(self):
        while True:
            row = 0
            column = 0
            for i in range(len(self.threads)):
                if row < 2:
                    self.background[:self.grid_height, self.grid_width*row:self.grid_width*(row+1)] = self.threads[i].frame
                    row += 1
                else:
                    self.background[self.grid_height:, self.grid_width*(row-2):self.grid_width*(row-2+1)] = self.threads[i].frame
                    row += 1
                ## Code insert model
                    self.threas[i].frame
                ##
                self.threads[i].run()

            cv2.imshow('Multi-Video Capture', self.background)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        for i in self.threads:
            self.threads.video.release()
        cv2.destroyAllWindows()

multi_video_capture = MultiVideoCapture(800, 600, './data/1.mp4', './data/2.mp4', './data/3.mp4', './data/4.mp4')
multi_video_capture.start()
multi_video_capture.display()
