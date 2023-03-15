import os
import shutil
import stat
import math
import json
import numpy as np
import cv2

def makedir(folder):
	if not os.path.exists(folder):
		try:
			os.makedirs(folder)
		except OSError:
			pass
	return folder

def delete_all(filePath):
	if os.path.exists(filePath):
		for fileList in os.walk(filePath):
			for name in fileList[2]:
				os.chmod(os.path.join(fileList[0],name), stat.S_IWRITE)
				os.remove(os.path.join(fileList[0],name))
			shutil.rmtree(filePath)
			return "delete ok"
		else:
			return "no filepath"


def save_single_intrinsics(mtx, dist, path):
	data={
		"K":mtx.tolist(),
		"D":dist.tolist()
	}
	with open(path,"w") as f:
		json.dump(data,f)

def save_stereo_intrinsics(path, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q,ret,ret1,ret2,lrpe,rrpe):
	data={
		"Left Re-Projection error":lrpe,
		"Right Re-Projection error":rrpe,
		"Left RMS":ret1,
		"Right RMS":ret2,
		"Stereo RMS":ret,
		"K1":K1.tolist(),
		"D1":D1.tolist(),
		"K2":K2.tolist(),
		"D2":D2.tolist(),
		"R":R.tolist(),
		"T":T.tolist(),
		"E":E.tolist(),
		"F":F.tolist(),
		"R1":R1.tolist(),
		"R2":R2.tolist(),
		"P1":P1.tolist(),
		"P2":P2.tolist(),
		"Q":Q.tolist()
	}
	with open(path,"w") as f:
		json.dump(data,f,indent=2)

def save_fisheyestereo_intrinsics(path, K1, D1, K2, D2, R, T, R1, R2, P1, P2, Q,ret,ret1,ret2,lrpe,rrpe):
	data={
		"Left Re-Projection error":lrpe,
		"Right Re-Projection error":rrpe,
		"Left RMS":ret1,
		"Right RMS":ret2,
		"Stereo RMS":ret,
		"K1":K1.tolist(),
		"D1":D1.tolist(),
		"K2":K2.tolist(),
		"D2":D2.tolist(),
		"R":R.tolist(),
		"T":T.tolist(),
		"R1":R1.tolist(),
		"R2":R2.tolist(),
		"P1":P1.tolist(),
		"P2":P2.tolist(),
		"Q":Q.tolist()
	}
	with open(path,"w") as f:
		json.dump(data,f,indent=2)
   
def load_intrinsics(path):
	with open(path) as f:
		data=json.load(f)
	for key,value in data.items():
		data[key]=np.array(value)
	return data

def load_extrinsics(path):
	with open(path, "r") as fp:
		camera_extrinsics = json.load(fp)
		yaw = float(camera_extrinsics["cam_yaw_at_ground"]) 
		pitch = float(camera_extrinsics["cam_pitch_at_ground"]) 
		roll = float(camera_extrinsics["cam_roll_at_ground"]) 
		tx = float(camera_extrinsics["cam_Tx_at_ground"]) 
		ty = float(camera_extrinsics["cam_Ty_at_ground"]) 
		tz = float(camera_extrinsics["cam_Tz_at_ground"]) 
		yaw = yaw * math.pi / 180.0
		pitch = pitch * math.pi / 180.0
		roll = roll* math.pi / 180.0

		c1 = math.cos(yaw) 
		c2 = math.cos(pitch)
		c3 = math.cos(roll)
		s1 = math.sin(yaw)
		s2 = math.sin(pitch)
		s3 = math.sin(roll)

		r = [[c1 * c3 - c2 * s1 * s3, c3 * s1 + c1 * c2 * s3, s2 * s3],
          	[-c1 * s3 - c2 * c3 * s1, c1 * c2 * c3 - s1 * s3, c3 * s2],
          	[s1 * s2, -c1 * s2, c2]]

		t = [tx, ty, tz]
	return np.asmatrix(r), np.asmatrix(t).T

