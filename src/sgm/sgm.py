import sys
sys.path.insert(0,'./src')
import numpy as np
import cv2
from interval3 import Interval
from utils import timer
class SemiGlobalMatching:
    def __init__(self,height,width) -> None:
        self.output_path='./output/demo/'
        self.min_disparity=10
        self.max_disparity=64
        self.width=width
        self.height=height
        self.P1 = 5
        self.P2= 70
        self.disp_range=self.max_disparity-self.min_disparity
    @timer
    def census(self,img,window_size=5):
        census  = np.zeros(shape=(self.height, self.width), dtype=np.uint64)
        offset=window_size//2
        for y in range(offset, self.height-offset):
            for x in range(offset, self.width-offset):
                center_pixel = img[y, x]
                reference = np.full(shape=(window_size, window_size), fill_value=center_pixel, dtype=np.int64)

                binary_pattern = []
                window_image = img[(y - offset):(y + offset + 1), (x - offset):(x + offset + 1)]

                # 比较矩形窗口其他像素和中心像素的大小
                comparison = window_image - reference
                for j in range(comparison.shape[0]):
                    for i in range(comparison.shape[1]):
                        if (i, j) != (offset, offset):
                            # 如果比中心像素小则编码为 1， 否则为 0
                            if comparison[j, i] < 0:
                                bit = 1
                            else:
                                bit = 0
                            binary_pattern.append(str(bit))

                binary_pattern = "".join(binary_pattern)

                decimal_number = int(binary_pattern, base=2)
                census[y, x] = decimal_number
        return census

    def hanming(self,x,y):
        xor = np.int64(np.bitwise_xor(x, y))
        xor = np.binary_repr(xor)
        distance = xor.count('1')
        return distance

    def normalize(self,disp):
        return 255.0 * disp / self.max_disparity
    @timer
    def cost_volume(self,left_census,right_census):
        disp_range=self.max_disparity-self.min_disparity
        left_cost_volume = np.zeros(shape=(self.height, self.width, disp_range), dtype=np.uint32)
        right_cost_volume = np.zeros(shape=(self.height, self.width, disp_range), dtype=np.uint32)

        # 为了便于理解，这里逐一遍历所有的像素点，代码复杂度高
        for i in range(self.height):
            for j in range(self.width):
                for ind,d in enumerate(range(self.min_disparity,self.max_disparity)):
                    # 从左图寻找右图的对应像素点，得到左图的代价
                    censusl_val=left_census[i,j]
                    rj = (j-d) if (j-d) > 0 else (self.max_disparity-d+j)
                    censusr_val=right_census[i,rj]
                    left_cost=self.hanming(censusl_val,censusr_val)
                    left_cost_volume[i,j,ind]=left_cost

                    # 从右图寻找左图对应的像素点，得到右图的代价
                    censusr_val=right_census[i,j]
                    rj = (j+d) if (j+d) < self.width else (self.width-self.max_disparity+(j-(self.width-d)))
                    censusl_val=left_census[i,rj]
                    right_cost=self.hanming(censusl_val,censusr_val)
                    right_cost_volume[i,j,ind]=right_cost

        return left_cost_volume,right_cost_volume

    def get_path_cost(self,slice,is_forward=False,offset=1):
        '''
            该段代码的理解，参见 代价聚合 一节中给出的示例解释
        '''
        slice= slice if not is_forward else np.flip(slice,axis=0)
        other_dim, disparity_dim = slice.shape
        disparities = [d for d in range(self.min_disparity,self.max_disparity)] * disparity_dim
        disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)
        penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=np.uint32)

        penalties[np.abs(disparities - disparities.T) == 1] = self.P1
        penalties[np.abs(disparities - disparities.T)  > 1] = self.P2

        minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.uint32)
        minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

        for i in range(offset, other_dim):
            previous_cost = minimum_cost_path[i - 1, :]
            current_cost = slice[i, :]
            costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
            costs = np.amin(costs + penalties, axis=0)
            minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
        return minimum_cost_path
    @timer
    def cost_aggregation(self,cost_volume):
        '''
        ↘ ↓ ↙
        → p ←
        ↗ ↑ ↖
        '''
        # 总的结果，8个方向 
        aggregation_volume=np.zeros((self.height,self.width,self.disp_range,8),dtype=np.uint32)
        # 缓存某个方向的结果，如向右
        path_aggregation = np.zeros((self.height, self.width, self.disp_range), dtype=np.uint32)
        # 缓存相反方向的结果
        forward_aggregation = np.zeros((self.height, self.width, self.disp_range), dtype=np.uint32)

        # (height,weight)的坐标作为元素值，分x，y分开存放
        ys=np.array([i for i in range(self.height)]*self.width).reshape(self.width,self.height).T
        xs=np.array([i for i in range(self.width)]*self.height).reshape(self.height,self.width)

        # 聚合水平方向
        def left_right_aggregation():
            # 遍历高度得每一个高度上的水平cost_volume
            for i in range(self.height):
                right=cost_volume[i,:,:] # (width,disp_range)
                right_aggregation=self.get_path_cost(right) # (width,disp_range)
                path_aggregation[i,:,:]= right_aggregation
                # 反向
                left_aggregation=self.get_path_cost(right,is_forward=True)
                forward_aggregation[i,:,:]=np.flip(left_aggregation,axis=0)
            aggregation_volume[:,:,:,0]=path_aggregation
            aggregation_volume[:,:,:,1]=forward_aggregation
        # 聚合竖直方向
        def up_down_aggregation():
            # 遍历宽度得每一个宽度上的竖直cost_volume
            for i in range(self.width):
                down=cost_volume[:,i,:] # (height,disp_range)
                down_aggregation=self.get_path_cost(down) # (height,disp_range)
                path_aggregation[:,i,:]= down_aggregation
                # 反向
                up_aggregation=self.get_path_cost(down,is_forward=True)
                forward_aggregation[:,i,:]=np.flip(up_aggregation,axis=0)
            aggregation_volume[:,:,:,2]=path_aggregation
            aggregation_volume[:,:,:,3]=forward_aggregation
        #从左上到右下对角线方向
        def dagonal_1_aggregation():
            # 通过np.diagonal方法取对角线元素，offset范围为(-height+1,width-1)
            for i in range(-self.height+1,self.width-1):
                right_down=cost_volume.diagonal(offset=i).T # (1,disp_rage) ~ (height,disp_rage)
                right_down_aggregation=self.get_path_cost(right_down)
                # 取与right_down对应的在(height,width)上的x和y的坐标
                ys_i=ys.diagonal(offset=i).T
                xs_i=xs.diagonal(offset=i).T
                path_aggregation[ys_i,xs_i,:]=right_down_aggregation
                # 反向
                left_up_aggregation=self.get_path_cost(right_down,is_forward=True)
                # 坐标也反向
                ys_i =np.flip(ys_i)
                xs_i =np.flip(xs_i)
                forward_aggregation[ys_i,xs_i,:]=np.flip(left_up_aggregation,axis=0)
            aggregation_volume[:,:,:,4]=path_aggregation
            aggregation_volume[:,:,:,5]=forward_aggregation
        #从右上到左下对角线方向
        def dagonal_2_aggregation():
            for i in range(-self.height+1,self.width-1):
                left_down=np.flipud(cost_volume).diagonal(offset=i).T # (1,disp_rage) ~ (height,disp_rage)
                left_down_aggregation=self.get_path_cost(left_down)
                # 取与left_down对应的在(height,width)上的x和y的坐标
                ys_i=ys.diagonal(offset=i).T
                xs_i=xs.diagonal(offset=i).T
                path_aggregation[ys_i,xs_i,:]=left_down_aggregation
                # 反向
                right_up_aggregation=self.get_path_cost(left_down,is_forward=True)
                # 坐标也反向
                ys_i =np.flip(ys_i)
                xs_i =np.flip(xs_i)
                forward_aggregation[ys_i,xs_i,:]=np.flip(right_up_aggregation,axis=0)
            aggregation_volume[:,:,:,6]=path_aggregation
            aggregation_volume[:,:,:,7]=forward_aggregation
        
        left_right_aggregation()
        up_down_aggregation()
        dagonal_1_aggregation()
        dagonal_2_aggregation()
        return aggregation_volume

    def get_disparity(self,aggregation_volume):
        # 对所有路径的代价求和
        volume = np.sum(aggregation_volume, axis=3)
        # 取代价和最小的index作为视差值
        disparity = np.argmin(volume, axis=2)
        # index要加上最小视差
        disparity = disparity + self.min_disparity
        return disparity

    def imwrite(self,img,_type):
        if 'cost_volume' in _type:
            img = np.argmin(img, -1).astype(np.uint8)+self.min_disparity
        img = self.normalize(img)
        cv2.imwrite(self.output_path+"{}.png".format(_type), img)
    @timer
    def left_right_check(self,disp_left,disp_right,threshold):
        disparity=np.zeros((self.height,self.width),dtype=np.int8)
        for j in range(self.height):
            for i in range(self.width):
                # d = x_l - x_r   =>   x_r = x_l -d
                displ = disp_left[j,i]
                x_r = i - displ
                if x_r in Interval(0,self.width,upper_closed=False):
                    dispr=disp_right[j,x_r]
                    if abs(displ-dispr)>threshold:
                        disparity[j,i]=-1
                    else:
                        disparity[j,i]=disp_left[j,i]
                else:
                    disparity[j,i]=-1
        return disparity
    @timer
    def fill_hole(self,disp):
        # 只添补大于最小视差的区域
        disp_fh=disp[:,self.min_disparity:]
        def newcoor(i,nb_coor):
            nb_coor=list(nb_coor)
            if i==0:
                nb_coor[0]=nb_coor[0]-1
                nb_coor[1]=nb_coor[1]-1
            elif i==1:
                nb_coor[1]=nb_coor[1]-1
            elif i==2:
                nb_coor[0]=nb_coor[0]+1
                nb_coor[1]=nb_coor[1]-1
            elif i==3:
                nb_coor[0]=nb_coor[0]-1
            elif i==4:
                nb_coor[0]=nb_coor[0]+1
            elif i==5:
                nb_coor[0]=nb_coor[0]-1
                nb_coor[1]=nb_coor[1]+1
            elif i==6:
                nb_coor[1]=nb_coor[1]+1
            elif i==7:
                nb_coor[0]=nb_coor[0]+1
                nb_coor[1]=nb_coor[1]+1
            return tuple(nb_coor)

        def findval(i,nb_coor):
            nb_coor = newcoor(i,nb_coor)
            while (nb_coor[0] in Interval(0,self.height-1) and nb_coor[1] in Interval(0,self.width-self.min_disparity-1)):
                d = disp_fh[nb_coor]
                if d >= 0:
                    return d
                nb_coor = newcoor(i,nb_coor)
            return None

        def neighborhood(coor):
            x,y=coor
            vals=[]
            # 8领域坐标
            nb_coors=[
                (x-1,y-1),
                (x,y-1),
                (x+1,y-1),
                (x-1,y),
                (x+1,y),
                (x-1,y+1),
                (x,y+1),
                (x+1,y+1)
            ]
            # 取8领域内的有效值的平均值来添补空洞
            for i,nb_coor in enumerate(nb_coors):
                if nb_coor[0] in Interval(0,self.height-1) and nb_coor[1] in Interval(0,self.width-self.min_disparity-1):
                    d = disp_fh[nb_coor]
                    if d >= 0:
                        vals.append(d)
                    # 该方向上的相邻值也是空洞值，则继续向外扩张训练新的有效值
                    else:
                        d=findval(i,nb_coor)
                        if d != None:
                            vals.append(d)
            return np.mean(vals)
        
        cnt=0
        # 为防止一次可能添补不全，循环计算直至所有的-1都被填充
        while -1 in disp_fh:
            cnt+=1
            hole_xs,hole_ys =  np.where(disp_fh == -1)
            print ("\r {}:{}".format(cnt,len(hole_xs)), end="")
            for hole in zip(hole_xs,hole_ys):
                fill_val=neighborhood(hole)
                disp_fh[hole]= fill_val
        disp[:,self.min_disparity:]=disp_fh
        return disp
    def match(self,imgl,imgr):
        # census变换
        census_left  = self.census(imgl)
        census_right = self.census(imgr)
        cv2.imwrite(self.output_path+"census_left.png", census_left.astype(np.uint8))
        cv2.imwrite(self.output_path+"census_right.png", census_right.astype(np.uint8))

        # 代价计算
        cost_volume_left,cost_volume_right=self.cost_volume(census_left,census_right)
        self.imwrite(cost_volume_left,'cost_volume_left')
        self.imwrite(cost_volume_right,'cost_volume_right')
       
        # 代价聚合
        aggregation_volume_left = self.cost_aggregation(cost_volume_left)
        aggregation_volume_right = self.cost_aggregation(cost_volume_right)

        # 视差计算
        disp_left = self.get_disparity(aggregation_volume_left)
        disp_right = self.get_disparity(aggregation_volume_right)
        self.imwrite(disp_left,'disp_left')
        self.imwrite(disp_right,'disp_right')

        # 左右一致性检查
        disp = self.left_right_check(disp_left,disp_right,threshold=3)
        self.imwrite(disp,'disp_checked')

        # 空洞填充
        disp = self.fill_hole(disp)
        self.imwrite(disp,'disp_fillholed')

        # 中值滤波
        disp_blured = cv2.medianBlur(np.uint8(disp[:,self.min_disparity:]),3)
        disp[:,self.min_disparity:]=disp_blured
        self.imwrite(disp,'disp_medianBlured')