from time import time
import os
import stat
import shutil
def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s time cost: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

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
