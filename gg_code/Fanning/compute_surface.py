
import nibabel as nib
import numpy as np
from dipy.tracking.metrics import spline
from dipy.tracking.streamline import set_number_of_points
from scilpy.utils.streamlines import load_in_voxel_space
from scilpy.tractanalysis.uncompress import uncompress
from scipy import ndimage as nd
from scipy import stats
from trimeshpy import TriMesh_Vtk


def binary_mask(input_trk, input_anat):

    streamlines = load_in_voxel_space(input_trk, input_anat)
    dim = input_anat.get_data().shape
    stl_vox = np.zeros((dim[0], dim[1], dim[2]), dtype=np.int32)
    indices = uncompress(streamlines)
    for stl in indices:
        for i in range(stl.shape[0]):
            x, y, z = stl[i]
            stl_vox[x, y, z] = 1

    stl_vox = nd.binary_fill_holes(stl_vox).astype(stl_vox.dtype)

    return stl_vox


def surf_from_phi(phi):
    volume = stats.threshold(phi, threshmin=np.percentile(phi, 0.95), newval=0)
    volume[:,:,0] = volume.max()
    return volume


def laplacian_smoothing(surface_file):
    mesh = TriMesh_Vtk(surface_file, None)

    vertices = mesh.laplacian_smooth(10, 1.0, l2_dist_weighted=False, area_weighted=False, backward_step=True,
                                     flow_file=None)
    mesh.set_vertices(vertices)
    return mesh