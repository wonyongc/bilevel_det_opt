#!/bin/bash
# Beam-pipe geometry + the FCC-ee compensating/screening solenoid field map
# (k4geo MDI_o1_ShapeBased_v01/fields_antisol_map.xml, map by M. Koratzinos, as used by the
# FCC-ee MDI background studies).  Adds to geom_scepcal_xtal_bp.xml -> geom_scepcal_xtal_bpas.xml,
# probes the total field along the axis and along the outgoing beam line, and runs a
# 2-event geantino load test.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
SRC=$RUN/geom_scepcal_xtal_bp.xml
DST=$RUN/geom_scepcal_xtal_bpas.xml
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
[ -s $SRC ] || { echo "missing $SRC (run make_bp_geom.sh first)"; exit 1; }
ls -la $K4GEO/fieldmaps/FCCee_antisolenoid_fieldmapXYZ_20180212.root || exit 1

sed '/BeamInstrumentation_o1_v01.xml" *\/>/a\  <include ref="${K4GEO}/FCCee/MDI/compact/MDI_o1_ShapeBased_v01/fields_antisol_map.xml" />' $SRC > $DST
echo "== includes and field length in the new file:"; grep -n "include ref" $DST; grep -o 'zmax="[^"]*"' $DST

echo "== field probe: Bz relative to the collision point, on the detector axis and on the outgoing beam line (x = 15 mrad * z)"
cd $RUN
python - "$DST" <<'PY' 2>&1 | grep -v "^DD4hep\|INFO\|WARN" | tail -20
import sys
try:
    import dd4hep
    d = dd4hep.Detector.getInstance(); d.fromXML(sys.argv[1]); f = d.field()
    b0 = f.magneticField(dd4hep.Position(0, 0, 0)).z()
    print(f"  Bz at the collision point (internal units): {b0:.4g}")
    for zm in (0.0, 0.5, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0):
        zc = zm * 100.0                                   # DD4hep lengths are in cm
        a = f.magneticField(dd4hep.Position(0, 0, zc)).z() / b0
        bl = f.magneticField(dd4hep.Position(0.015 * zc, 0, zc))
        print(f"  z = {zm:3.1f} m   axis Bz/B0 = {a:6.3f}   beam line Bz/B0 = {bl.z()/b0:6.3f}  |Bperp|/B0 = {(bl.x()**2+bl.y()**2)**0.5/abs(b0):6.3f}")
except Exception as e:
    print("  field probe failed (not fatal):", e)
PY

echo "== 2-event geantino test"
ddsim --compactFile $DST --enableGun --gun.particle geantino --gun.energy "10*GeV" \
      --gun.distribution uniform -N 2 --outputFile $HOME/bpas_test.edm4hep.root > $HOME/bpas_test.log 2>&1
echo "beam-pipe + compensating-field geometry: ddsim exit code $?"
grep -i -E "error|exception|fieldxyz|antisol" $HOME/bpas_test.log | head -10
