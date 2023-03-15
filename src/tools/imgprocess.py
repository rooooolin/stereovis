
import cv2
import numpy as np
import os
from tools.utils import load_intrinsics,makedir,delete_all
class ImgProcess:
    def __init__(self,camera,outpath,intrinsics_folder,is_calib=False) -> None:
        self.camera=camera
        self.outpath=outpath
        self.intrinsics_folder=intrinsics_folder
        # load maps
        lmapx_file = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "lmapx.xml"), cv2.FILE_STORAGE_READ)
        lmapy_file = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "lmapy.xml"), cv2.FILE_STORAGE_READ)
        rmapx_file = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "rmapx.xml"), cv2.FILE_STORAGE_READ)
        rmapy_file = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "rmapy.xml"), cv2.FILE_STORAGE_READ)
        self.lmapx = lmapx_file.getNode("data").mat()
        self.lmapy = lmapy_file.getNode("data").mat()
        self.rmapx = rmapx_file.getNode("data").mat()
        self.rmapy = rmapy_file.getNode("data").mat()
        if not is_calib:
            self.cam_intrinsics=load_intrinsics(self.intrinsics_folder+self.camera+'/params.json')

    def rectify_plot(self,limg: np.ndarray,rimg: np.ndarray,img_name: str):
        h,w,_=limg.shape
        rr=np.zeros((h,w*2,3),np.uint8)
        rr[0:h,0:w]=limg
        rr[0:h,w:w*2]=rimg
        for i in range(1,10):
            interval=h//10
            ptStart = (0, interval*i)
            ptEnd = (w*2, interval*i)			
            cv2.line(rr, ptStart, ptEnd, (0, 0, 255), 2, 4)
        cv2.imwrite(self.outpath+'/rectify_'+img_name,rr)

    def rectify(self,img: np.ndarray,index: int):
        height, width, channel = img.shape
        if isinstance(index, list):
            return cv2.initUndistortRectifyMap(index[0], index[1], index[2], index[3], (width, height), cv2.CV_32FC1)
 
        else:
            mapx,mapy=cv2.initUndistortRectifyMap(self.cam_intrinsics['K'+str(index)], 
                self.cam_intrinsics['D'+str(index)],self.cam_intrinsics['R'+str(index)],
                self.cam_intrinsics['P'+str(index)], (width, height), cv2.CV_32FC1)
            return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    
    def get_ud_imgs(self, limg, rimg):
        ud_left = cv2.remap(limg, self.lmapx, self.lmapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        ud_right = cv2.remap(rimg, self.rmapx, self.rmapy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return ud_left, ud_right

    def get_rectify_map(self,img, paras):
        mapx,mapy=self.rectify(img, paras)
        cv_file_x = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "mapx.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_x.write("data", mapx)
        cv_file_x.release()
        cv_file_y = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "mapy.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_y.write("data", mapy)
        cv_file_y.release()

    def get_stereo_rectify_map(self, camera_type, img_size, limg_params, rimg_params):
        if camera_type == "stereo":
            lmapx, lmapy = cv2.initUndistortRectifyMap(limg_params[0], limg_params[1], limg_params[2], limg_params[3], img_size, cv2.CV_32FC1)
            rmapx, rmapy  = cv2.initUndistortRectifyMap(rimg_params[0], rimg_params[1], rimg_params[2], rimg_params[3], img_size, cv2.CV_32FC1)
        elif camera_type == "fisheye_stereo":
            lmapx, lmapy = cv2.fisheye.initUndistortRectifyMap(limg_params[0], limg_params[1], limg_params[2], limg_params[3], img_size, cv2.CV_32FC1)
            rmapx, rmapy  = cv2.fisheye.initUndistortRectifyMap(rimg_params[0], rimg_params[1], rimg_params[2], rimg_params[3], img_size, cv2.CV_32FC1)

        cv_file_x = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "lmapx.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_x.write("data", lmapx)
        cv_file_x.release()
        cv_file_y = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "lmapy.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_y.write("data", lmapy)
        cv_file_y.release()
        cv_file_x = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "rmapx.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_x.write("data", rmapx)
        cv_file_x.release()
        cv_file_y = cv2.FileStorage(os.path.join(self.intrinsics_folder+self.camera, "rmapy.xml"), cv2.FILE_STORAGE_WRITE)
        cv_file_y.write("data", rmapy)
        cv_file_y.release()
        return lmapx, lmapy, rmapx, rmapy

    def stereo_seg(self,img):
        h,w=img.shape[:2]
        mid=w//2
        limg=img[0:h, 0:mid]
        rimg=img[0:h,mid:w]
        return limg,rimg

    def imseg(self,basedir):
        imgs=os.listdir(basedir+'raw/')
        delete_all(basedir+'/LEFT')
        delete_all(basedir+'/RIGHT')
        makedir(basedir+'/LEFT')
        makedir(basedir+'/RIGHT')
        for img in imgs:
            imgname=img
            img=basedir+'raw/'+img
            limg,rimg=self.stereo_seg(cv2.imread(img))
            cv2.imwrite(basedir+'/LEFT/'+imgname,limg)
            cv2.imwrite(basedir+'/RIGHT/'+imgname,rimg)