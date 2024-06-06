from scipy.ndimage import binary_fill_holes, binary_closing
import trimesh
from scipy.ndimage import distance_transform_edt
import numpy as np

def cube_voxel(indices, cube_size):
    '''converts a matrix of indecies (2d np.ndarray) showing the occupied voxels 
        into a standard cubic n*n*n by scaling.
    
    indeces  : np.ndarray: This shows the occupancy
                            where row values shows that the acutal shapee's space with 
                            that index is occupied.
                
    cube_size: int       : can be 32, 64, 128 etc. for convenience use values 
                            of power 2.
    '''
    cube_size = cube_size-1 #(ex. 256-->max will be 255)
    coef_cubicize = (indices.max(axis=0)-indices.min(axis=0))/cube_size

    cube_voxels = (indices-indices.min(axis=0))//coef_cubicize
    cube_voxels = cube_voxels.astype(int)
    return cube_voxels
    
    
    
    
def save_xyz(voxel_grid, filename=None, sdf_mode='neg'):
    '''given a voxel_grid of size n*m*q it saves values with 'sdf_mode' values'''
    
    
    if sdf_mode==["neg", "n", "negative", -1, "-", "-1"]:
    	indices = np.argwhere(voxel_grid <= 0)
    elif sdf_mode==["pos", "p", "positive", +1, "+", "1", "+1"]:
    	indices = np.argwhere(voxel_grid > 0)
    else:
     	indices = np.argwhere(voxel_grid <= 0)
    		
    if (filename==None):
        filename = 'output.xyz'
    
    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write each point to the file
        for point in indices:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f'{filename} is saved...')
    
    
def mesh_to_cubid_sdf(file_name,resolution=32, cube_size=50):
    mesh = trimesh.load(file_name)
    
    # Convert the mesh to a voxel grid
    voxel_data = mesh.voxelized(pitch=mesh.extents.max() / resolution, method='subdivide')#.matrix
    voxel_data = voxel_data.matrix.astype(int)
    # +
    
    occupancy_cube_vox = np.zeros((cube_size, cube_size, cube_size), dtype=np.int8) # 0 or 1 in n*n*n 3d matrix
    indices = np.argwhere(voxel_data > 0.5) 


    std_indices = cube_voxel(indices, cube_size)

    for index in std_indices:
        occupancy_cube_vox[index[0], index[1], index[2]]=1   
    voxel_data = occupancy_cube_vox
    
    
    
    voxel1_ =  distance_transform_edt(voxel_data) 
    voxel2_ =  distance_transform_edt((voxel_data+1)%2)
    
    # sdf_voxel1 = voxel1_ #- voxel2_
    # sdf_voxel = voxel2_
    sdf_voxel = -voxel1_ + voxel2_

    return sdf_voxel          
