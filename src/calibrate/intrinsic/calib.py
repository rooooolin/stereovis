
import numpy as np
import cv2
import glob
from tools.utils import save_single_intrinsics,save_stereo_intrinsics,save_fisheyestereo_intrinsics
import os
import shutil

class Calibration(object): 
	
	def __init__(self,chessboard_w,chessboard_h,imgbase,camera_type,camera,size,savedir,imgP):
		self.chessboard_w=chessboard_w
		self.chessboard_h=chessboard_h
		
		self.criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
		self.camera_type=camera_type
		self.camera=camera
		self.size=size
		self.imgbase=imgbase
		self.savedir=savedir
		self.imgP=imgP
		self.cnt=0

	def get_points(self):
		objp = np.zeros((self.chessboard_h*self.chessboard_w,3), np.float32)
		objp[:,:2] = np.mgrid[0:self.chessboard_w, 0:self.chessboard_h].T.reshape(-1, 2)
		objp=objp*self.size
		objpoints = []
		left_imgpoints = []  
		right_imgpoints = [] 

		left_images = glob.glob(self.leftpath + '*.jpg')
		self.mapxy_img=left_images[0]
		right_images = glob.glob(self.rightpath + '*.jpg')
		left_images.sort()
		right_images.sort()
	
		for index,(limg, rimg) in enumerate(zip(left_images, right_images)):
			right = cv2.imread(rimg)
			gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
			# find left corners
			ret_right, corners_right = cv2.findChessboardCorners(gray_right, (self.chessboard_w, self.chessboard_h),
																cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

		
			left = cv2.imread(limg)
			gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
			# find right corners
			ret_left, corners_left = cv2.findChessboardCorners(gray_left, (self.chessboard_w, self.chessboard_h),
															cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

			# Check for the flipped checkerboard!
			# diff = corners_left - corners_right
			# lengths = np.linalg.norm(diff[:, :, 1], axis=-1)
			# sum = np.sum(lengths, axis=0)
			# if (sum > 2000.0):
			# 	print("THIS STEREO PAIR IS BROKEN!!! Diff is: "+str(sum))
			# 	corners_right = np.flipud(corners_right)


			# both left and right have corner
			if ret_left and ret_right:
				self.cnt+=1
				objpoints.append(objp)
				
				corners2_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), self.criteria)
				ret_r = cv2.drawChessboardCorners(right, (9, 6), corners_right, ret_right)
				right_imgpoints.append(corners2_right)
			
				corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), self.criteria)
				ret_l = cv2.drawChessboardCorners(left, (9, 6), corners_left, ret_left)
				left_imgpoints.append(corners2_left)
			
				# for check manually
				width,height=gray_left.shape[::-1]
				rr=np.zeros((height,width*2,3),np.uint8)
				rr[0:height,0:width]=ret_l
				rr[0:height,width:width*2]=ret_r
				cv2.imwrite(self.imgbase+'/corners/'+os.path.basename(rimg),rr)


			else:
				print(f"[{index+1}/{len(left_images)}]Chessboard couldn't detected. Image pair: LEFT/{os.path.basename(limg)} and RIGHT/{os.path.basename(rimg)}")
				#os.remove(self.imgbase+self.camera+'/chessboard/raw/'+os.path.basename(limg))
				shutil.move(self.imgbase+'/raw/'+os.path.basename(limg),self.imgbase+'/raw_failed_find_corners/'+os.path.basename(limg))
				continue
		print(f"effective images: {self.cnt}")
		return objpoints,left_imgpoints,right_imgpoints,gray_left,gray_right
	
	def re_projection_error(self,objpoints,imgpoints,mtx,dist,rvecs,tvecs):
		# MSE and RMS error https://blog.csdn.net/qq_32998593/article/details/113063216
		mean_error = 0
		for i in range(len(objpoints)):
			if self.camera_type == "stereo":
				imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			elif self.camera_type == "fisheye_stereo":
				imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

			error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			mean_error += error
		rpe=mean_error/len(objpoints)
		return rpe

	def check_undistorted_img(self, lmapx, lmapy, rmapx, rmapy):
		left_images = glob.glob(self.leftpath + '*.jpg')
		right_images = glob.glob(self.rightpath + '*.jpg')
		left_images.sort()
		right_images.sort()
		for img_lm, img_r in zip(left_images, right_images):
			img = cv2.imread(img_lm)
			undistorted_img = cv2.remap(img, lmapx, lmapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
			cv2.imwrite(self.imgbase+'/undistorted_LEFT/'+os.path.basename(img_lm),undistorted_img)
			img = cv2.imread(img_r)
			undistorted_img = cv2.remap(img, rmapx, rmapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
			cv2.imwrite(self.imgbase+'/undistorted_RIGHT/'+os.path.basename(img_r),undistorted_img)	

	def stereo(self):
		self.leftpath=self.imgbase+'/LEFT/'
		self.rightpath=self.imgbase+'/RIGHT/'
		objpoints,left_imgpoints,right_imgpoints,gray_left,gray_right=self.get_points()
		# calibrate left camera =>left K,D
		flags=0
		flags |= cv2.CALIB_TILTED_MODEL # 标定函数使用倾斜传感器模型并返回14个系数，如果不设置标志，则函数计算并返回只有5个失真系数。
		#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
		#flags |= cv2.CALIB_THIN_PRISM_MODEL
		flags |= cv2.CALIB_RATIONAL_MODEL # 计算k4，k5，k6三个畸变参数
		ret1, K1, D1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, left_imgpoints, gray_left.shape[::-1], None,None,flags=flags,criteria=self.criteria)
		lrpe=self.re_projection_error(objpoints,left_imgpoints,K1, D1, rvecs1, tvecs1)
		print(f"Left camera calibrate rms: {round(ret1,4)}; ReProjectionError: {round(lrpe,4)}")
		# calibrate right camera =>right K,D
		ret2, K2, D2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, right_imgpoints, gray_right.shape[::-1], None,None,flags=flags,criteria=self.criteria)
		rrpe=self.re_projection_error(objpoints,right_imgpoints,K2, D2, rvecs2, tvecs2)
		print(f"Right camera calibrate rms: {round(ret2,4)}; ReProjectionError: {round(rrpe,4)}")

		# calibrate stereo =>stereo R,T
		flags = 0
		flags |= cv2.CALIB_TILTED_MODEL
		flags |= cv2.CALIB_FIX_INTRINSIC
		flags |= cv2.CALIB_RATIONAL_MODEL
		#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
		#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
		#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
		ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, K1, D1, K2, D2, gray_left.shape[::-1], criteria=self.criteria,flags=flags)
		print(f"Stereo calibration rms: {round(ret,4)}")
		
		# stereo rectify => left R,P  right R,P  stereo Q
		R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, gray_left.shape[::-1], R, T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)

		save_stereo_intrinsics(self.savedir+'/params.json', K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q,ret,ret1,ret2,lrpe,rrpe)
		# self.imgP.get_rectify_map(cv2.imread(self.mapxy_img), [K1,D1,R1,P1])
		lmapx, lmapy, rmapx, rmapy = self.imgP.get_stereo_rectify_map(self.camera_type, gray_left.shape[::-1], [K1,D1,R1,P1], [K2,D2,R2,P2])
		self.check_undistorted_img(lmapx, lmapy, rmapx, rmapy)

		print(f"result store in:{self.savedir}")

	def fisheye_stereo(self):
		# https://stackoverflow.com/questions/50857278/raspicam-fisheye-calibration-with-opencv
		self.leftpath=self.imgbase+'/LEFT/'
		self.rightpath=self.imgbase+'/RIGHT/'
		objpoints,left_imgpoints,right_imgpoints,gray_left,gray_right=self.get_points()
		
		N_OK = len(left_imgpoints)
		K1 = np.zeros((3, 3))
		D1 = np.zeros((4, 1))
		K2 = np.zeros((3, 3))
		D2 = np.zeros((4, 1))
		R = np.zeros((1, 1, 3), dtype=np.float64)
		T = np.zeros((1, 1, 3), dtype=np.float64)
		rvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
		tvecs1 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
		rvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
		tvecs2 = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
		
		objp = np.zeros( (self.chessboard_w*self.chessboard_h, 1, 3) , np.float64)
		objp[:,0, :2] = np.mgrid[0:self.chessboard_w, 0:self.chessboard_h].T.reshape(-1, 2)
		objpoints = np.asarray(objpoints, dtype=np.float64)
		left_imgpoints = np.asarray(left_imgpoints, dtype=np.float64)
		right_imgpoints = np.asarray(right_imgpoints, dtype=np.float64)
		objpoints = np.reshape(objpoints, (N_OK, 1, self.chessboard_w*self.chessboard_h, 3))
		left_imgpoints = np.reshape(left_imgpoints, (N_OK, 1, self.chessboard_w*self.chessboard_h, 2))
		right_imgpoints = np.reshape(right_imgpoints, (N_OK, 1, self.chessboard_w*self.chessboard_h, 2))

		# calibrate left camera =>left K,D
		flags=0
		flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC # 每次优化迭代内参后，重新计算外参
		flags |= cv2.fisheye.CALIB_FIX_SKEW            # 偏差系数alpha会被置成0，并一直保持在0的状态
		flags |= cv2.fisheye.CALIB_CHECK_COND          # 函数将检查条件数和合法性
		# flags |= cv2.fisheye.CALIB_FIX_INTRINSIC
		ret1, K1, D1, rvecs1, tvecs1 = cv2.fisheye.calibrate(objpoints, left_imgpoints, gray_left.shape[::-1],None,None,rvecs1,tvecs1,flags=flags,criteria=self.criteria)
		lrpe=self.re_projection_error(objpoints,left_imgpoints,K1, D1, rvecs1, tvecs1)
		print(f"Left camera calibrate rms: {round(ret1,4)}; ReProjectionError: {round(lrpe,4)}")
		# calibrate right camera =>right K,D
		ret2, K2, D2, rvecs2, tvecs2 = cv2.fisheye.calibrate(objpoints, right_imgpoints, gray_right.shape[::-1],None,None,rvecs2,tvecs2,flags=flags,criteria=self.criteria)
		rrpe=self.re_projection_error(objpoints,right_imgpoints,K2, D2, rvecs2, tvecs2)
		print(f"Right camera calibrate rms: {round(ret2,4)}; ReProjectionError: {round(rrpe,4)}")


		flags=0
		flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
		flags |= cv2.fisheye.CALIB_FIX_SKEW
		flags |= cv2.fisheye.CALIB_CHECK_COND
		ret, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(objpoints, left_imgpoints, right_imgpoints, 
			 K1, D1, K2, D2, (1280, 480), R, T, flags=flags, criteria=self.criteria)
		print(f"fisheye_stereo calibration rms: {round(ret,4)}")

		# stereo rectify => left R,P  right R,P  stereo Q
		# balance: 在最大焦距值和最小焦距值范围之间设置新焦距值
		# fov_scale: 新焦距的除数
		R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1, D1, K2, D2, 
													gray_left.shape[::-1], R, T, 
													flags=cv2.CALIB_ZERO_DISPARITY,
													balance= 0, fov_scale=1.0)

		# save params
		save_fisheyestereo_intrinsics(self.savedir+'/params.json', K1, D1, K2, D2,
		 	R, T, R1, R2, P1, P2, Q, ret, ret1,ret2,lrpe,rrpe)

		# # get Undistort map
		# # take a look of this https://gist.github.com/mesutpiskin/0412c44bae399adf1f48007f22bdd22d
		# dim2=None
		# dim3=None
		# balance=1
		# DIM = gray_right.shape[::-1] 
		# dim1 = gray_right.shape[::-1]  # dim1 is the dimension of input image to un-distort
		# assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
		# if not dim2:
		# 	dim2 = dim1
		# if not dim3:
		# 	dim3 = dim1
		# scaled_K = K1 * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
		# scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
		# 	# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
		# new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D1, dim2, np.eye(3), balance=balance)
		# lmapx, lmapy = cv2.fisheye.initUndistortRectifyMap(scaled_K, D1, np.eye(3), new_K, dim3, cv2.CV_32FC1)
		lmapx, lmapy, rmapx, rmapy = self.imgP.get_stereo_rectify_map(self.camera_type, gray_left.shape[::-1], [K1,D1,R1,P1], [K2,D2,R2,P2])
		# check maps
		self.check_undistorted_img(lmapx, lmapy, rmapx, rmapy)

		print(f"result store in:{self.savedir}")

	def single(self):
		self.leftpath=self.rightpath=self.imgbase
		
		objpoints,imgpoints,_,gray,_=self.get_points()
		ret, k, d, r, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		
		save_single_intrinsics(k, d, self.savedir+'/params.json')
		print(f"Calibration is finished. RMS: {round(ret,4)}")
		print(f"result store in:{self.savedir}/params.json")

	def run(self):
		if self.camera_type == "stereo":
			self.stereo()
		elif self.camera_type == "fisheye_stereo":
			self.fisheye_stereo()
		else:
			self.single()

