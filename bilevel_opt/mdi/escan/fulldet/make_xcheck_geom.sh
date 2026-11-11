#!/bin/bash
# Cross-check geometries for the crystal background of the full IDEA (Z 1x).
# The full IDEA gives 30 % less crystal energy than the standalone baseline (barrel -13 %, endcaps -44 %).
# Beam pipe, beam instrumentation, LumiCal and field are the SAME in both (Beampipe_o4_v05,
# BeamInstrumentation_o1_v01, LumiCal_o1_v01, antisolenoid map, detector field to 3010 mm; the CAD-based
# MDI is commented out in k4geo IDEA_o2_v01.xml). What differs is the detector around the crystals and the
# steering. Three runs separate the causes:
#   A  geom_scepcal_xtal_lumi.xml (standalone baseline) with SteeringFile_fulldet.py -> steering effect
#   B  geom_idea_nodr_map.xml     full IDEA without dual-readout tubes and muon system -> what is behind
#   C  geom_idea_nodrabs_map.xml  B without the 4.2 mm lead end-plate absorber          -> the lead
# This script writes B and C from $RUN/geom_idea_map.xml and load-tests all three (2 geantinos each).
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
SRC=$RUN/geom_idea_map.xml
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
[ -s $SRC ] || { echo "missing $SRC (DET=IDEA bash fulldet/make_fulldet_geom.sh)"; exit 1; }
[ -s $RUN/geom_scepcal_xtal_lumi.xml ] || { echo "missing $RUN/geom_scepcal_xtal_lumi.xml"; exit 1; }
cd $RUN

python - $SRC $RUN <<'PY' || exit 1
import re, sys
src, run = sys.argv[1:3]
t = open(src).read()
def drop(t, names):
    for n in names:
        t, k = re.subn(r'[ \t]*<include ref="[^"]*/%s"\s*/>[ \t]*\n?' % re.escape(n), "", t)
        if k != 1:
            sys.exit(f"  {n}: {k} active include lines found, expected 1 - nothing written")
        print(f"  removed {n}")
    return t
b = drop(t, ["DREndcapTubes_o1_v01.xml", "DRBarrelTubes_o1_v01.xml", "MuonSystem_o1_v01.xml"])
open(f"{run}/geom_idea_nodr_map.xml", "w").write(b); print("wrote geom_idea_nodr_map.xml")
c = drop(b, ["EndPlateAbsorber_o1_v01.xml"])
open(f"{run}/geom_idea_nodrabs_map.xml", "w").write(c); print("wrote geom_idea_nodrabs_map.xml")
PY

for g in geom_scepcal_xtal_lumi geom_idea_nodr_map geom_idea_nodrabs_map; do
  echo "== $g: detectors"
  python - $RUN/$g.xml <<'PY' 2>&1 | grep "^@@" | sed 's/^@@//'
import sys, dd4hep
d = dd4hep.Detector.getInstance(); d.fromXML(sys.argv[1])
names = sorted(p.first for p in d.detectors())
print("@@  " + " ".join(names))
PY
  echo "== $g: 2-geantino load test"
  /usr/bin/time -f "   wall %e s, max memory %M kB" ddsim --compactFile $RUN/$g.xml --enableGun --gun.particle geantino \
      --gun.energy "10*GeV" --gun.distribution uniform -N 2 --outputFile $HOME/xchk_test.edm4hep.root > $HOME/xchk_$g.log 2>&1
  echo "   ddsim exit code $?"; tail -1 $HOME/xchk_$g.log
  grep -i -E "error|exception|not defined|undefined|fatal" $HOME/xchk_$g.log | grep -v -i "G4Exception-START\|warning" | head -5
done
