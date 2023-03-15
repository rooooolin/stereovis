import os

import argparse

from utils import *
from tools.cam import WebCam
from tools.depth import Depth,Disparity
from tools.imgprocess import ImgProcess
# import gi
# gi.require_version('Gtk', '2.0')

from tools.plots import Plots
from tools.reconstruct import AxesTrans
import glob
import cv2
import numpy as np

# import faulthandler
# faulthandler.enable()


basedir='./'
def get_args():
	parser = argparse.ArgumentParser(description='main function')
	parser.add_argument('--camera', type=str, default='GSLSM22100_1280x480', help='') # 
	parser.add_argument('--outpath', type=str, default=basedir+'output/test-0923/', help='')
	parser.add_argument('--intrinsics', type=str, default=basedir+'data/parameters/intrinsics/', help='')
	parser.add_argument('--extrinsics', type=str, default=basedir+'data/parameters/extrinsics/BDR_rice_0413/', help='')
	parser.add_argument('--camid', type=int, default=0, help='')
	parser.add_argument('--cam_display', type=bool, default=True, help='')
	parser.add_argument('--image', type=str, default='/media/xiao/DATADRIVE1/projects/stereovis/data/datasets/BDR_rice_0413/materials/1.jpg', help='')

	# parser.add_argument('--cam_display', type=bool, default=False, help='')
	# parser.add_argument('--image', type=str, default="/home/mzh/PycharmProjects/4-Stereo/stereo_vision_project/output/test-materials/captured_20220410195754107148.jpg", help='')
	args = parser.parse_args()
	return args

class StereoVis:
	def __init__(self):
		#self.basedir=os.getcwd()
		self.args=get_args()
		self.camera=self.args.camera
		self.cam = WebCam(self.args.camid, resolution=self.args.camera.split('_')[-1])
		self.imgP=ImgProcess(self.camera,self.args.outpath,self.args.intrinsics)
		self.depth=Depth(self.imgP)
		self.disparity=Disparity(self.imgP)
		self.axesT=AxesTrans(self.args.extrinsics,self.args.camera)
		
		self.plt=Plots()

	def get_depth(self, limg: np.ndarray,rimg: np.ndarray,filename: str, dist_check=False,dem=False):
	
		disp,lm,limg=self.disparity.sgbm_matcher(limg,rimg,filename) #stereo_match
		disp=disp[:,lm:]
		cv2.imwrite(self.args.outpath+'disp_'+filename,disp)

		#get depth by reprojectImageTo3D
		point_3d,depth_map=self.depth.reproject(disp)
		cv2.imwrite(self.args.outpath+'depth_'+filename,depth_map*100)
		
		# if need depth map visualize
		if dist_check:
			limg=limg[:,lm:]
			self.plt.distance_check(limg,depth_map,self.args.outpath,filename, 25)
		
		# if need create digital elevation map
		if dem:
			self.plt.dem(disp,self.args.outpath,filename)

		return disp,depth_map,point_3d
	
	def calculation(self,limg,rimg,filename,dist_check=True,dem=False):
		disp, depth_map,point_3d=self.get_depth(limg,rimg,filename,dist_check=dist_check,dem=dem)
		# project_to_world coord
		world_point_3d = self.axesT.project_to_world(point_3d)
		# delete the points if  (x < 0 and x > width ; y < 0 and y > length ; z < 0 and z > height)
		for i,upper in enumerate([1.0, 1.0, 1.0]):    
			world_point_3d=np.delete(world_point_3d, np.where((world_point_3d[:,i]<-1.0) | (world_point_3d[:,i]>upper))[0], 0)
		world_point_3d=np.delete(world_point_3d, np.where((world_point_3d[:,1]>-0.2) & (world_point_3d[:,1]<0))[0], 0)
		world_point_3d[:,1]=-world_point_3d[:,1]
		
		
		np.savetxt(self.args.outpath+filename+"_world_point_3d.txt",world_point_3d)
	



	def perform(self, frame: np.ndarray,filename:str, task: str):
		if task == "R":
			self.cmd='R'
			self.msg='calculating..., please wait'
			self.cam.save(frame, self.args.outpath, filename)
			limg,rimg=self.imgP.stereo_seg(frame)
			self.calculation(limg,rimg,filename)
			self.msg='done! result stored in '+ self.args.outpath
				
		elif task == "S":
			self.cmd='S'
			self.cam.save(frame, self.args.outpath, filename)
			self.msg='Captured '+filename+' successfully! stored in '+ self.args.outpath

	def display(self):
		start_time = time.time()
		counter = 0
		self.fps=0
		self.cmd=''
		self.msg=''
		while True:
			frame = self.cam.get_frame()
			filename=list(frame.keys())[0]
			frame= list(frame.values())[0]
	
			self.clone = frame.copy()
			self.clone = self.cam.display_buttons(self.clone)

			counter += 1
			if (time.time() - start_time) > 1:
				self.fps= counter / (time.time() - start_time)
				counter = 0
				start_time = time.time()

			self.clone = self.cam.init_display(self.clone,str(round(self.fps,2)))
			self.clone = self.cam.msg_display(self.clone,self.cmd,self.msg,(255, 255, 255))
			
			event = self.cam.show(self.clone)
			task = self.cam.check_event(event)
			self.perform(frame, filename, task)

	def get_image(self):
		# if None => open camera to capture frame
		if not self.args.image:
			self.cam.on()
			if self.args.cam_display:
				self.display()
			else:
				frame=self.cam.get_frame(self.args.outpath)
				return frame
		# if folder =>  get all images in directory
		if os.path.isdir(self.args.image):
			images={}
			files=glob.glob(self.args.image + '*.jpg')
			for f in files:
				filename=os.path.basename(f)
				images[filename]=cv2.imread(f)
			return images
		# if path => get only one image
		else:
			return {os.path.basename(self.args.image):cv2.imread(self.args.image)}

	def run(self):
		print('clean up image cache ')
		delete_all(self.args.outpath)
		makedir(self.args.outpath)

		images=self.get_image()
		print(f'image number:{len(images)}')

		print('image process..')
		for filename,image in images.items():
			filename=filename.replace('.jpg','.png')
			limg,rimg=self.imgP.stereo_seg(image)
			self.calculation(limg,rimg,filename)

		print('all done!')

def main():
	sv=StereoVis()
	sv.run()
	

if __name__=='__main__':
    main()
