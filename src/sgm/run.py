

from sgm import SemiGlobalMatching
import cv2
def run():
    imgl  = cv2.imread('./data/demo/left.png',cv2.IMREAD_GRAYSCALE)
    imgr = cv2.imread('./data/demo/right.png',cv2.IMREAD_GRAYSCALE)
    height,width=imgl.shape[:2]
    sgm=SemiGlobalMatching(height,width)
    sgm.match(imgl,imgr)
if __name__=='__main__':
    run()