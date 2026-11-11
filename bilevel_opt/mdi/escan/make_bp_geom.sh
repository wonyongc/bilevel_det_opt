#!/bin/bash
# Two geometry variants of the crystal-only standalone geometry, then a 2-event geantino test:
#   geom_scepcal_xtal_field.xml : solenoid field extended to 1.2 x coil half length (2.616 m),
#                                 as in k4geo IDEA_o1_v03; the standalone file stops it at
#                                 2.18 m, short of the endcap face (~2.3 m)
#   geom_scepcal_xtal_bp.xml    : the same field + the IDEA beam pipe (k4geo MDI_o1_v00, as
#                                 included by IDEA_o1_v03): AlBeMet/paraffin centre, gold coating,
#                                 copper downstream pipes, SR masks, tungsten shields,
#                                 compensating/screening solenoids
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
SRC=$RUN/geom_scepcal_xtal_theta_width-1cm.xml
FLD=$RUN/geom_scepcal_xtal_field.xml
DST=$RUN/geom_scepcal_xtal_bp.xml
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

echo "== constants the beam pipe needs, found in the dimensions file (each should be >= 1):"
DIM=$(grep -o 'include ref="[^"]*_dimensions.xml"' $SRC | cut -d'"' -f2)
for c in CrossingAngle CentralBeamPipe_rmax CentralBeamPipe_zmax BPWWall BPWCool SeparatedBeamPipe_z \
         SeparatedBeamPipe_rmax ConeBeamPipe_Rmax BeamPipeWidthFirstCone MiddleOfSRMask_z SynchRadMaskSize \
         QD0_min_z QD0_max_z LumiCal_max_z BeamPipeTantalShieldWidth BeamPipeGoldWidth CompSol_min_z; do
  printf "  %-28s %s\n" $c $(grep -c "name=\"$c\"" $DIM)
done

sed 's|zmax="Solenoid_Coil_half_length"|zmax="1.2 * Solenoid_Coil_half_length"|' $SRC > $FLD
sed '/_dimensions.xml" *\/>/a\  <include ref="${K4GEO}/FCCee/MDI/compact/MDI_o1_v00/Beampipe_o4_v05.xml" />\n  <include ref="${K4GEO}/FCCee/MDI/compact/MDI_o1_v00/BeamInstrumentation_o1_v01.xml" />' $FLD > $DST
echo "== field line in the new files (should say 1.2 * Solenoid_Coil_half_length):"; grep -h -o 'zmax="[^"]*"' $FLD $DST
echo "== includes in the beam-pipe file:"; grep -n "include ref" $DST

echo "== 2-event geantino tests (geometry loads, no missing constants or materials)"
cd $RUN
ddsim --compactFile $FLD --enableGun --gun.particle geantino --gun.energy "10*GeV" \
      --gun.distribution uniform -N 2 --outputFile $HOME/fld_test.edm4hep.root > $HOME/fld_test.log 2>&1
echo "field-only geometry: ddsim exit code $?"
ddsim --compactFile $DST --enableGun --gun.particle geantino --gun.energy "10*GeV" \
      --gun.distribution uniform -N 2 --outputFile $HOME/bp_test.edm4hep.root > $HOME/bp_test.log 2>&1
echo "beam-pipe geometry:  ddsim exit code $?"
grep -i -E "error|exception|unknown|not defined|undefined|fatal" $HOME/bp_test.log | head -15
grep -i -E "BeBeampipe|BeamPipe|SynchRadMask|CompSol|ScreenSol" $HOME/bp_test.log | head -8
