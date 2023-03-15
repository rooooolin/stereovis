import os
import sys


sys.path.insert(0,'./python/')
import argparse
from calib import Calibration
from tools.imgprocess import ImgProcess
from tools.utils import *

# http://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

def get_args():
	parser = argparse.ArgumentParser(description='Camera calibration')
	parser.add_argument('--imgbase', type=str, default='/home/mzh/PycharmProjects/4-Stereo/stereo_vision_project/data/datasets/JZY_166_0410/chessboard/', help='chessboard images path')
	parser.add_argument('--chessboard-w', type=int, default=9, help='number of grids')
	parser.add_argument('--chessboard-h', type=int,  default=6,help='number of grids')
	parser.add_argument('--camera', type=str, default='GSLSM22100_1280x480', help='camera brand')
	parser.add_argument('--camera_type', type=str, default='stereo', help='single, stereo, fisheye_stereo')
	parser.add_argument('--size', type=float, default=24, help='mm')
	args = parser.parse_args()
	return args

def run():
	args=get_args()
	imgP=ImgProcess(args.camera,None,'./data/parameters/intrinsics/',True)
	imgP.imseg(args.imgbase)
	cali=Calibration(
		chessboard_w=args.chessboard_w,
		chessboard_h=args.chessboard_h,
		imgbase=args.imgbase,
		camera_type=args.camera_type,
		camera=args.camera,
		size=args.size,
		savedir='./data/parameters/intrinsics/'+args.camera,
		imgP=imgP
	)

	delete_all(args.imgbase+'/corners/')
	makedir(args.imgbase+'/corners/')
	delete_all(args.imgbase+'/undistorted_LEFT/')
	makedir(args.imgbase+'/undistorted_LEFT/')
	delete_all(args.imgbase+'/undistorted_RIGHT/')
	makedir(args.imgbase+'/undistorted_RIGHT/')

	if not os.path.exists(args.imgbase+'/raw_failed_find_corners/'):
		makedir(args.imgbase+'/raw_failed_find_corners/')

	cali.run()

if __name__=='__main__':
	
	run()