module purge
module use @MODULE_ROOT@
module load @DETECTOR_MODULE@

source @SPACK_SETUP@
spack env activate @SPACK_ENV_DIR@

source @VIEW_ROOT@/bin/thisroot.sh
source @VIEW_ROOT@/bin/thisdd4hep.sh

export LD_LIBRARY_PATH=@DETECTOR_INSTALL_LIB@:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=@DETECTOR_INSTALL_INCLUDE@:$ROOT_INCLUDE_PATH
