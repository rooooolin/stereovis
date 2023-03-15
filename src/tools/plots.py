
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2


class Plots:
    def __init__(self) -> None:
        pass

    def dem(self,depth_map,outpath,filename,wsample=100,hsample=50):
        
        height,width=depth_map.shape[:2]

        def get_verts_edges():
            verts=[];edges=[]
            windexs=list(map(int,np.linspace(start = 0, stop = width-1, num = wsample).tolist()))
            hindexs=list(map(int,np.linspace(start = 0, stop = height-1, num = hsample).tolist()))

            for i,windex in enumerate(windexs):
                for j,hindex in enumerate(hindexs):
                    verts.append([i-wsample/2,j-hsample/2,depth_map[hindex,windex],1])
           
            hs=np.array((range(hsample)))*wsample
            for hi in range(len(hs)-1):
                for w in range(wsample-1):
                    edges.append([hs[hi]+w,hs[hi]+w+1])
                    edges.append([hs[hi]+w,hs[hi+1]+w])
                edges.append([hs[hi]+w+1,hs[hi+1]+w+1])
            for w in range(wsample-1):
                edges.append([hs[-1]+w,hs[-1]+w+1])

            return verts,edges

        def get_camera(verts,paras):
            offset = np.array([[1, 0, 0, paras['offsetX']],
                            [0, -1, 0, paras['offsetY']],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

            P = np.array([[(paras['f'] * width) / (2 * paras['Px']), paras['skew'], 0, 0],
                        [0, (paras['f'] * height) / (2 * paras['Py']), 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])

            C = np.array([[1, 0, 0, -paras['Cx']],
                        [0, 1, 0, -paras['Cy']],
                        [0, 0, 1, -paras['Cz']],
                        [0, 0, 0, 1]])

            Rx = np.array([[1, 0, 0, 0],
                        [0, np.cos(paras['RotX']), - np.sin(paras['RotX']), 0],
                        [0, np.sin(paras['RotX']), np.cos(paras['RotX']), 0],
                        [0, 0, 0, 1]])

            Ry = np.array([[np.cos(paras['RotY']), 0, np.sin(paras['RotY']), 0],
                        [0, 1, 0, 0],
                        [- np.sin(paras['RotY']), 0, np.cos(paras['RotY']), 0],
                        [0, 0, 0, 1]])

            Rz = np.array([[np.cos(paras['RotZ']), - np.sin(paras['RotZ']), 0, 0],
                        [np.sin(paras['RotZ']), np.cos(paras['RotZ']), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

            G = np.array([[1, 0, 0, -paras['Gx']],
                        [0, 1, 0, -paras['Gy']],
                        [0, 0, 1, -paras['Gz']],
                        [0, 0, 0, 1]])

            x = [0] * len(verts)

            for i in range(len(verts)):
                x[i] = np.matmul(G, np.array(verts[i]))
                x[i] = np.matmul(Rz, x[i])
                x[i] = np.matmul(Ry, x[i])
                x[i] = np.matmul(Rx, x[i])
                x[i] = np.matmul(C, x[i])
                x[i] = np.matmul(P, x[i])

                N = np.array([[1 / x[i][2], 0, 0, 0],
                            [0, 1 / x[i][2], 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

                x[i] = np.matmul(N, x[i])
                x[i] = np.matmul(offset, x[i])

            return x
        def render(camera, edges):
            canvas = 255 * np.ones((640, 480, 3), dtype='uint8')
            for edge in edges:
                try:
                    cv2.line(canvas,
                        (int(camera[edge[0]][0]), int(camera[edge[0]][1])),
                        (int(camera[edge[1]][0]), int(camera[edge[1]][1])),
                        color=(0, 0, 255),
                        thickness=1)
                except:continue
            return canvas
        
        '''
        'RotX' : 0,
            'RotY' : 2.86, 
            'RotZ' : 3.14,

        '''
        paras={
            'Gx' : 0,
            'Gy' : 0,
            'Gz' : 0,

            'RotX' : 3.14,
            'RotY' : 3.0,
            'RotZ' : 0,

            'Cx' : 0,
            'Cy' : 0,
            'Cz' : 5,

            'f' : 0.05,
            'Px' : 0.06,
            'Py' : 0.048,

            'offsetX' : width/2,
            'offsetY' : height/2,
            'skew' : 0
        }
        
        verts,edges=get_verts_edges()
        camera = get_camera(verts,paras)
        cv2.imwrite(outpath+'dem_'+filename,render(camera, edges))

        # while True:
        #     camera = get_camera(verts,paras)
        #     cv2.imshow('render', render(camera, edges))
        #     print(paras['RotZ'])
        #     paras['RotZ'] += 0.01
        #     #paras['RotZ'] -= 0.005
        #     cv2.waitKey(0)
      
    def distance_check(self,img: np.ndarray,depth: np.ndarray,outpath,filename,interval=50):
        height, width,_ = img.shape
        numrows, numcols = height//interval,width//interval
     
        for col in range(1,numcols+1):
            cv2.line(img,(col*interval,0),(col*interval,height),color=(0, 0, 255),thickness=1)
        for row in range(1,numrows+1):
            cv2.line(img,(0,row*interval),(width,row*interval),color=(0, 0, 255),thickness=1)

        for row in range(1,numrows):
            for col in range(1,numcols):
                x=col*interval
                y=row*interval
                # z=round(depth[y-interval//2,x-interval//2],2) #mid
                z=round(depth[y,x],2) 
                #z=round(depth[(y1+interval)//2,(x1+interval)//2]/1000,2)

                cv2.putText(img, str(z), (x, y),cv2.FONT_HERSHEY_SIMPLEX, interval/80.0, (255, 255, 255), 1)
        cv2.imwrite(outpath+'distance_'+filename,img)

    def points_3d(self,points,outpath,filename,width,length,height):
        fig=plt.figure()
        ax= fig.add_subplot(111,projection='3d')
        x=points[:,0]
        y=points[:,1]
        z=points[:,2]
        ax.scatter(x,y,z,s=0.1)
        # ax.set_xlim(0,width)
        # ax.set_ylim(0,length)
        # ax.set_zlim(0,height)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(outpath+filename+'.png')



        
