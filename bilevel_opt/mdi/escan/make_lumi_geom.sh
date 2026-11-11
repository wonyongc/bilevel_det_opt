#!/bin/bash
# bpas3 (beam pipe + compensating/screening field map, detector field to 3010 mm)
# + the IDEA_o1_v03 LumiCal (k4geo FCCee/IDEA/compact/IDEA_o1_v03/LumiCal_o1_v01.xml):
#   W-Si calorimeter, 25 x 3.5 mm tungsten, r 55-112 mm, z 1074-1186.5 mm, on the
#   outgoing beam axis, + instrumentation/cooling rings out to r ~141 mm + 3.5 mm W back shield.
# Constants, vis attributes and materials the LumiCal needs but the standalone geometry
# lacks are added (IDEA_o1_v03 values) in lumical_extra.xml; detector IDs that would clash
# are moved.  Then a 2-event geantino load test.  Writes geom_scepcal_xtal_lumi.xml.
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
SRC=$RUN/geom_scepcal_xtal_bpas3.xml
DST=$RUN/geom_scepcal_xtal_lumi.xml
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
LCAL=$K4GEO/FCCee/IDEA/compact/IDEA_o1_v03/LumiCal_o1_v01.xml
[ -s $SRC ] || { echo "missing $SRC"; exit 1; }
[ -s $LCAL ] || { echo "missing $LCAL"; exit 1; }
cd $RUN

echo "== what the standalone geometry already has, and what gets added:"
python - $SRC $LCAL $RUN <<'PY' 2>&1 | grep -E "^@@|Traceback|Error" | head -30 | sed 's/^@@//'
import sys, re, ROOT, dd4hep
src, lcal, run = sys.argv[1:4]
d = dd4hep.Detector.getInstance(); d.fromXML(src)
def has(kind, n):
    try:
        if kind == "const": d.constant(n)
        elif kind == "mat": d.material(n)
        elif kind == "vis":
            return d.visAttributes(n).isValid()
        return True
    except Exception:
        return False
consts = [("world_side", "6100*mm"), ("env_safety", "0.1*mm"),
          ("LumiCal_max_z", "1186.5*mm"), ("LumiCal_min_z", "1074*mm"),
          ("LumiCal_dz", "(LumiCal_max_z-LumiCal_min_z)/2.0"),
          ("LumiCal_inner_radius", "55.0*mm"), ("LumiCal_outer_radius", "112.0*mm- env_safety"),
          ("LumiCal_Instr_thickness", "20*mm"), ("LumiCal_Instr_inner_radius", "LumiCal_outer_radius"),
          ("LumiCal_Instr_outer_radius", "LumiCal_outer_radius+LumiCal_Instr_thickness - env_safety"),
          ("LumiCal_Cool_thickness", "9.75*mm"), ("LumiCal_Cool_inner_radius", "LumiCal_Instr_outer_radius"),
          ("LumiCal_Cool_outer_radius", "LumiCal_Instr_outer_radius+LumiCal_Cool_thickness"),
          ("LumiCal_Shield_inner_radius", "LumiCal_inner_radius"),
          ("LumiCal_Shield_outer_radius", "LumiCal_outer_radius+LumiCal_Instr_thickness+LumiCal_Cool_thickness"),
          ("LumiCal_shield_dz", "1.75*mm")]
for n in ("LumiCal_max_z", "CrossingAngle", "CompSol_min_z", "SeparatedBeamPipe_z"):
    if has("const", n):
        print(f"@@  existing {n:22} = {d.constantAsDouble(n):g}  (DD4hep units: cm / rad)")
# detector IDs: keep IDEA's 15-18 unless a detector already uses one of them
used = set()
try:
    for p in d.detectors():
        try: used.add(int(p.second.id()))
        except Exception: pass
except Exception:
    pass
ids = [15, 16, 17, 18] if not used & {15, 16, 17, 18} else [115, 116, 117, 118]
print(f"@@  detector IDs in use: {sorted(used)} -> LumiCal IDs {ids}")
for n, v in zip(("DetID_LumiCal", "DetID_LumiCalInstrumentation", "DetID_LumiCalCooling", "DetID_LumiCalBackShield"), ids):
    consts.append((n, str(v)))
add_c = [(n, v) for n, v in consts if not has("const", n)]
vis = {"LCALVis": 'r="0.25" g="0.88" b="0.81"', "LCALInstrVis": 'r="0.35" g="0.0" b="0.47"',
       "LCALCoolVis": 'r="0.2" g="0.6" b="0"', "LCALShieldVis": 'r="0.5" g="0.5" b="0.5"',
       "SupportVis": 'r="0.2" g="0.2" b="0.2"', "BCLayerVis1": 'r="0.0" g="0.0" b="1.0"',
       "BCLayerVis2": 'r="0.0" g="1.0" b="0.0"', "BCLayerVis3": 'r="1.0" g="0.0" b="0.0"',
       "SeeThrough": 'r="1" g="1" b="1"'}
add_v = [n for n in vis if not has("vis", n)]
mats = {"TungstenDens24": '<D value="17.8" unit="g/cm3"/><fraction n="0.93" ref="W"/><fraction n="0.061" ref="Ni"/><fraction n="0.009" ref="Fe"/>',
        "Kapton": '<D value="1.43" unit="g/cm3"/><composite n="22" ref="C"/><composite n="10" ref="H"/><composite n="2" ref="N"/><composite n="5" ref="O"/>'}
add_m = [n for n in mats if not has("mat", n)]
for n in ("Silicon", "Copper", "Aluminum", "Air"):
    if not has("mat", n):
        print(f"@@  WARNING: material {n} missing from the standalone geometry")
print(f"@@  adding constants: {[n for n, _ in add_c] or 'none'}")
print(f"@@  adding vis: {add_v or 'none'};  adding materials: {add_m or 'none'}")
x = ["<lccdd>", "  <define>"] + [f'    <constant name="{n}" value="{v}"/>' for n, v in add_c] + ["  </define>"]
if add_m:
    x += ["  <materials>"] + [f'    <material name="{n}">{mats[n]}</material>' for n in add_m] + ["  </materials>"]
if add_v:
    x += ["  <display>"] + [f'    <vis name="{n}" alpha="1.0" {vis[n]} showDaughters="true" visible="true"/>' for n in add_v] + ["  </display>"]
open(f"{run}/lumical_extra.xml", "w").write("\n".join(x + ["</lccdd>", ""]))
t = open(lcal).read()
if has("const", "LumiCal_cell_size"):
    t = re.sub(r'<constant name="LumiCal_cell_size"[^>]*/>', "", t)
open(f"{run}/LumiCal_o1_v01_xtal.xml", "w").write(t)
print("@@  wrote lumical_extra.xml and LumiCal_o1_v01_xtal.xml")
PY
[ -s $RUN/lumical_extra.xml ] && [ -s $RUN/LumiCal_o1_v01_xtal.xml ] || { echo "generation failed"; exit 1; }

sed "/fields_antisol_map.xml\" *\/>/a\  <include ref=\"$RUN/lumical_extra.xml\" />\n  <include ref=\"$RUN/LumiCal_o1_v01_xtal.xml\" />" $SRC > $DST
echo "== includes and field length in $DST:"; grep -n "include ref" $DST; grep -o 'zmax="[^"]*"' $DST

echo "== LumiCal placement check"
python - $DST <<'PY' 2>&1 | grep -E "^@@|Traceback|Error" | head -30 | sed 's/^@@//'
import sys, ROOT, dd4hep
d = dd4hep.Detector.getInstance(); d.fromXML(sys.argv[1])
for n in ("LumiCal", "LumiCalInstrumentation", "LumiCalCooling", "LumiCalBackShield"):
    try:
        de = d.detector(n); print(f"@@  {n:24} built, id {de.id()}")
    except Exception as e:
        print(f"@@  {n:24} MISSING ({e})")
nav = ROOT.gGeoManager
for z in (1080, 1130, 1185, 1188):
    x = 0.015 * z + 70.0                                  # mm, 70 mm off the outgoing beam axis
    node = nav.FindNode(x / 10.0, 0.0, z / 10.0)
    print(f"@@  point z = {z} mm, 70 mm from the outgoing beam: {node.GetVolume().GetName() if node else None}"
          f" ({node.GetMedium().GetMaterial().GetName() if node else ''})")
PY

echo "== 2-event geantino test"
ddsim --compactFile $DST --enableGun --gun.particle geantino --gun.energy "10*GeV" \
      --gun.distribution uniform -N 2 --outputFile $HOME/lumi_test.edm4hep.root > $HOME/lumi_test.log 2>&1
rc=$?
echo "LumiCal geometry: ddsim exit code $rc"
grep -i -E "error|exception|not defined|undefined|fatal" $HOME/lumi_test.log | grep -v -i "G4Exception-START\|warning" | head -10
exit $rc
