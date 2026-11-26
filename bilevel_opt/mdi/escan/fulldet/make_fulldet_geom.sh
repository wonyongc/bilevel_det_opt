#!/bin/bash
# Build a full-detector geometry for the energy-scan background runs:
#   DET=CLD      k4geo FCCee/CLD/compact/CLD_o2_v08        (Si-W ECAL, steel-scintillator HCAL)
#   DET=ALLEGRO  k4geo FCCee/ALLEGRO/compact/ALLEGRO_o1_v03 (noble-liquid ECAL, TileCal HCAL)
#   DET=IDEA     k4geo FCCee/IDEA/compact/IDEA_o2_v01      (the COMPLETE IDEA with the SCEPCal crystals:
#                beam pipe, LumiCal, vertex, drift chamber, silicon wrapper, solenoid, dual-readout
#                tubes, muon system). The copy in ~/bilevel_det_opt/detector_config/IDEA_o2_v01 has
#                everything except the crystals commented out and is NOT the full detector.
#                With FIELD=map the solenoid field region is extended from 1.2 x coil half length
#                (2616 mm) to SOLZMAX (default 3010 mm, where the screening field of the map ends),
#                exactly as in the standalone baseline; SOLZMAX=native keeps the original.
#   FIELD=map    (default) add the FCC-ee compensating/screening-solenoid field map
#                (MDI_o1_ShapeBased_v01/fields_antisol_map.xml) as in the standalone baseline;
#                for CLD the two analytic "CompensatingSolenoid" fields of CLD_o2_v08 are removed
#                so the map is not double-counted
#   FIELD=native keep the detector's own field description unchanged
# Writes $RUN/geom_<det>_<field>.xml (all includes made absolute), probes Bz on the axis and on
# the outgoing beam line, lists the calorimeter readouts (= the hit-collection names the inner
# loop will read), and runs a 2-event geantino load test.
#
#   DET=CLD bash make_fulldet_geom.sh
#   DET=ALLEGRO bash make_fulldet_geom.sh
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DET=${DET:?set DET to CLD|ALLEGRO|IDEA}
FIELD=${FIELD:-map}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo

case $DET in
  CLD)     SRC=$K4GEO/FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml ;;
  ALLEGRO) SRC=$K4GEO/FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml ;;
  IDEA)    SRC=$K4GEO/FCCee/IDEA/compact/IDEA_o2_v01/IDEA_o2_v01.xml ;;
  *) echo "bad DET=$DET"; exit 1 ;;
esac
MAP=$K4GEO/FCCee/MDI/compact/MDI_o1_ShapeBased_v01/fields_antisol_map.xml
DST=$RUN/geom_$(echo $DET | tr A-Z a-z)_${FIELD}.xml
[ -s $SRC ] || { echo "missing $SRC"; exit 1; }
[ -s $MAP ] || { echo "missing $MAP"; exit 1; }
ls -la $K4GEO/fieldmaps/FCCee_antisolenoid_fieldmapXYZ_20180212.root || exit 1
cd $RUN

python - "$SRC" "$DST" "$MAP" "$FIELD" "$DET" "${SOLZMAX:-3010}" <<'PY'
import os, re, sys
src, dst, mapxml, field, det, solzmax = sys.argv[1:7]
base = os.path.dirname(os.path.abspath(src))
t = open(src).read()

# 1. make every relative file reference absolute (include, gdmlFile, ...), so the copy can live in $RUN
def absref(m):
    ref = m.group(1)
    if ref.startswith(("$", "/")) or not ref.endswith(".xml"):
        return m.group(0)
    return m.group(0).replace(ref, os.path.normpath(os.path.join(base, ref)))
t = re.sub(r'\bref="([^"]+)"', absref, t)

# 2. field handling
removed = []
if field == "map":
    if det == "CLD":
        for name in ("CompensatingSolenoidZplus", "CompensatingSolenoidZminus"):
            t, n = re.subn(r'<field name="%s"[^>]*>\s*</field>' % name, "", t, flags=re.S)
            removed.append((name, n))
    if det == "IDEA" and solzmax != "native":
        # the detector solenoid must reach to where the screening solenoid of the map ends, or the
        # screening field is left uncompensated between the two (standalone baseline: 3010 mm)
        t, n = re.subn(r'(<field name="GlobalSolenoid"[^>]*?zmax=")[^"]*(")', r'\g<1>%s*mm\2' % solzmax, t, flags=re.S)
        removed.append(("GlobalSolenoid zmax -> %s mm" % solzmax, n))
    if "fields_antisol_map.xml" not in t:
        t = t.replace("</lccdd>", '  <include ref="%s" />\n</lccdd>' % mapxml)
open(dst, "w").write(t)
print("wrote", dst)
for name, n in removed:
    print("  field change, %s: %d place(s)" % (name, n))
print("  file references:")
for ref in re.findall(r'\bref="([^"]+\.xml)"', t):
    print("    ", ref)
PY
[ -s $DST ] || { echo "generation failed"; exit 1; }

echo "== field probe: Bz relative to the collision point, on the axis and on the outgoing beam line"
python - "$DST" <<'PY' 2>&1 | grep -v "^DD4hep\|INFO\|WARN" | tail -20
import sys
try:
    import dd4hep, ROOT
    P = ROOT.dd4hep.Position          # the python binding has no dd4hep.Position
    d = dd4hep.Detector.getInstance(); d.fromXML(sys.argv[1]); f = d.field()
    b0 = f.magneticField(P(0, 0, 0)).z()
    print(f"  Bz at the collision point (internal units): {b0:.4g}")
    for zm in (0.0, 0.5, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 3.5):
        zc = zm * 100.0
        a = f.magneticField(P(0, 0, zc)).z() / b0
        bl = f.magneticField(P(0.015 * zc, 0, zc))
        print(f"  z = {zm:3.1f} m   axis Bz/B0 = {a:6.3f}   beam line Bz/B0 = {bl.z()/b0:6.3f}  |Bperp|/B0 = {(bl.x()**2+bl.y()**2)**0.5/abs(b0):6.3f}")
except Exception as e:
    print("  field probe failed (not fatal):", e)
PY

echo "== calorimeter readouts (hit-collection names) and detector ids"
python - "$DST" <<'PY' 2>&1 | grep -E "^@@" | sed 's/^@@//'
import sys, dd4hep
d = dd4hep.Detector.getInstance(); d.fromXML(sys.argv[1])
try:
    for r in d.readouts():
        print(f"@@  readout {r.first}")
except Exception as e:
    print("@@  readouts() failed:", e)
try:
    for p in d.detectors():
        try: print(f"@@  detector {p.first:28} id {p.second.id()}")
        except Exception: pass
except Exception as e:
    print("@@  detectors() failed:", e)
PY

echo "== 2-event geantino test"
ddsim --compactFile $DST --enableGun --gun.particle geantino --gun.energy "10*GeV" \
      --gun.distribution uniform -N 2 --outputFile $HOME/fulldet_${DET}_test.edm4hep.root > $HOME/fulldet_${DET}_test.log 2>&1
rc=$?
echo "$DET geometry ($FIELD field): ddsim exit code $rc"
grep -i -E "error|exception|not defined|undefined|fatal|fieldxyz|antisol" $HOME/fulldet_${DET}_test.log | grep -v -i "G4Exception-START\|warning" | head -10
exit $rc
