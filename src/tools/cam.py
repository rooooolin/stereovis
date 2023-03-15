
import cv2
import numpy as np
import datetime
import time

# cv2.namedWindow("Frame")

class WebCam:
    def __init__(self,cam_id,resolution='1280x480'):
        self.cap = None
        self.cam_id=cam_id
        self.resolution=resolution
        self.width,self.height=list(map(int,self.resolution.split('x')))
        
        
    def on(self):
        self.cap = cv2.VideoCapture(self.cam_id)

        #GXLSM22100 480p:1280*480 720p:2560*720  zed:3840*1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
       
        ret, frame = self.cap.read() 
   
        if not ret:
            print(f"open camera error")
            quit()
        else:
            print(f"open camera successfully")

    def get_frame(self,path=None):
        frame = self.cap.read()[1]

        filename = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f').replace('-','').replace(':','').replace('.','').replace(' ','')+ ".jpg"
        if path:
            self.save(frame,path,filename)
        return {filename:frame}

    def save(self,frame,path,filename):
        path=path+'/captured_'+filename
        cv2.imwrite(path,frame)

    def init_display(self, frame: np.ndarray, fps:str):
        cv2.putText(frame, f"FPS:{fps}, {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}", (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (30, 180, 30), 2)
        return frame

    def display_buttons(self, frame):

        cv2.putText(frame, "R -> Run Volume", (10, 80),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (30, 180, 30), 2)
        cv2.putText(frame, "S -> Only Save Image", (10, 110),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (215, 80, 80), 2)
        cv2.putText(frame, "Q -> Quit", (10, 140),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.7, (215, 80, 215), 2)
        return frame

    def msg_display(self, frame: np.ndarray, event: str,msg: str, color_mode: tuple):
        cv2.putText(frame, f"[[{event}]]:  {msg}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color_mode, 2)
        return frame

    def check_event(self, event):
        if event == ord("r"):
            return "R"

        elif event == ord("s"):
            return "S"

        elif event == ord("q"):
            self.off()
 
    def off(self):
        cv2.destroyAllWindows()
        self.cap.release()

    def show(self, frame: np.ndarray):
        if self.resolution!='1280x480':
            frame=cv2.resize(frame,(self.width//2,self.height//2))
        cv2.imshow("Frame", frame)
        return cv2.waitKey(25)
