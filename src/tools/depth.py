import cv2
import numpy as np




class Disparity:
    def __init__(self,imgP) -> None:
        self.imgP=imgP

    def sgbm_matcher(self,limg: np.ndarray,rimg: np.ndarray,img_name: str):
        '''
            SGBM based
            input: stereo images
            output: disparity map
        '''
        limg,rimg=self.imgP.get_ud_imgs(limg,rimg)
        self.imgP.rectify_plot(limg,rimg,img_name)
        
        l_grayimg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
        r_grayimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        
        block_size=9 #7
        min_disparity=0
        num_disparity=4*16
        left_matcher = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparity,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=32
            )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        ldisp=left_matcher.compute(l_grayimg,r_grayimg)
        rdisp=right_matcher.compute(r_grayimg,l_grayimg)
        ldisp = ldisp.astype(np.float32) / 16.
        rdisp = rdisp.astype(np.float32) / 16.
        
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.3)
        disp=wls_filter.filter(ldisp,l_grayimg,None,rdisp)
        
        
        return disp,min_disparity+num_disparity,limg
   

class Depth:
    def __init__(self,imgP):
        self.imgP=imgP

    def reproject(self,disparity: np.ndarray,scale=1.0):
        points_3d = cv2.reprojectImageTo3D(disparity, self.imgP.cam_intrinsics['Q'], handleMissingValues=True)
        points_3d=points_3d/1000.0
        depth_map = points_3d[:, :, 2]
        depth_map = depth_map * scale
        depth_map=depth_map.astype(np.float32)
        return np.asarray(points_3d, dtype=np.float32),depth_map

    




