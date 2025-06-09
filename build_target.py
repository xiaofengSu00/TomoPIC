import numpy as np
import pandas as pd
import mrcfile
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import map_coordinates

class TargetBuilder():
    def __init__(self):
        
        self.remove_flag = False 
    def generate_with_shapes(self,coords_data, tomo_data,ref):
        target_array = np.zeros(tomo_data.shape,dtype=np.int16)

        dim = tomo_data.shape
        for p in range(len(coords_data)):

            x = int(coords_data['x'].iloc[p])
            y = int(coords_data['y'].iloc[p])
            z = int(coords_data['z'].iloc[p])
            
            centeroffset = np.int(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[2] + x - centeroffset #+1
            y_vox = obj_voxels[1] + y - centeroffset #+1
            z_vox = obj_voxels[0] + z - centeroffset #+1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds
                    if self.remove_flag:
                        target_array[zz, yy, xx] = 0
                    else:
                        target_array[zz, yy, xx] = p + 1

        print('data occupancy finished')
        return np.int16(target_array)
    
    def generate_with_spheres(self, coords_data, tomo_data, radius):
        
        
        dim = [2*radius, 2*radius, 2*radius]
        
        ref = self.create_sphere(dim=dim, R = radius)
        target_array = self.generate_with_shapes(coords_data, tomo_data, ref)
        return target_array
    
    def  create_sphere(self,dim, R): 
        C = np.floor((dim[0]/2, dim[1]/2, dim[2]/2))
        x,y,z = np.meshgrid(range(dim[0]),range(dim[1]),range(dim[2]))

        sphere = ((x - C[0])/R)**2 + ((y - C[1])/R)**2 + ((z - C[2])/R)**2
        sphere = np.int8(sphere<=1)
        return sphere
    
    def rotate_array(self,array, orient):
        phi = orient[0]
        psi = orient[1]
        the = orient[2]

        # Some voodoo magic so that rotation is the same as in pytom:
        new_phi = -phi
        new_psi = -the
        new_the = -psi

        # create meshgrid
        dim = array.shape
        ax = np.arange(dim[0])
        ay = np.arange(dim[1])
        az = np.arange(dim[2])
        coords = np.meshgrid(ax, ay, az)

        # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
        xyz = np.vstack([coords[0].reshape(-1) - float(dim[0]) / 2,  # x coordinate, centered
                        coords[1].reshape(-1) - float(dim[1]) / 2,  # y coordinate, centered
                        coords[2].reshape(-1) - float(dim[2]) / 2])  # z coordinate, centered

        # create transformation matrix: the convention is not 'zxz' as announced in TOM toolbox
        r = R.from_euler('YZY', [new_phi, new_psi, new_the], degrees=True)
        ##r = R.from_euler('ZXZ', [the, psi, phi], degrees=True)
        mat = r.as_matrix()

        # apply transformation
        transformed_xyz = np.dot(mat, xyz)

        # extract coordinates
        x = transformed_xyz[0, :] + float(dim[0]) / 2
        y = transformed_xyz[1, :] + float(dim[1]) / 2
        z = transformed_xyz[2, :] + float(dim[2]) / 2

        x = x.reshape((dim[1],dim[0],dim[2]))
        y = y.reshape((dim[1],dim[0],dim[2]))
        z = z.reshape((dim[1],dim[0],dim[2])) # reason for strange ordering: see next line

        # the coordinate system seems to be strange, it has to be ordered like this
        new_xyz = [y, x, z]

        # sample
        arrayR = map_coordinates(array, new_xyz, order=1)

        return arrayR
