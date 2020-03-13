import nibabel as nib
import numpy as np
from dipy.reconst.shm import sh_to_sf
from dipy.core.sphere import Sphere
# from dipy.data import get_sphere
from gg_code.utils.visu import visu_odf
import sys

def main():
    odf_file = sys.argv[1]
    bvec_file = sys.argv[2]
    basis = sys.argv[3]

    odf = nib.load(odf_file)
    odf_sh = odf.get_fdata()
    bvec = np.loadtxt(bvec_file)
    bvec = bvec[1:, :]

    sph_gtab = Sphere(xyz=np.vstack(bvec))
    print(odf_sh.shape)

    odf_sf = sh_to_sf(odf_sh, sph_gtab, basis_type=basis, sh_order=8)
    visu_odf(bvec, odf_sf)


if __name__ == "__main__":
    main()