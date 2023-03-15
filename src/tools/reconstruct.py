
from tools.utils import load_extrinsics
import numpy as np


class BiDepth:
    def __init__(self) -> None:
        pass

class AxesTrans:
    def __init__(self,extrinsics,camera) -> None:
        self.extrinsics=extrinsics
        self.r, self.c = load_extrinsics(self.extrinsics+camera+'.json')	

    def project_to_world(self, point_3d: np.ndarray):
        
        world_point_3d = []
        # Pc = R*Pw + t, 其中 t = -R^-1 * c
        # world_point_3d = Pw = R^-1 * Pc + c
        world_point_3d = np.dot(self.r.I, (point_3d.reshape((-1,3)).T)) + self.c
        
        return world_point_3d.T