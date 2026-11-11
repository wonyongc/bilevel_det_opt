#!/bin/bash
# Short test of one full detector before production: 2 background crossings (Z, 1x) and 5 photons of
# 1 GeV through the production steering file. Prints the time per event and the hit collections.
#   DET=IDEA bash fulldet/test_fulldet.sh          (login node; a few minutes, mostly geometry start-up)
RUN=/pscratch/sd/s/sapta/bilevel/runs/smoke_test_z_pole
DET=${DET:?set DET to CLD|ALLEGRO|IDEA}; FIELD=${FIELD:-map}
DTAG=$(echo $DET | tr A-Z a-z)
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD ROOTSYS CPATH LIBRARY_PATH VIRTUAL_ENV 2>/dev/null || true
source /global/cfs/cdirs/m4956/sapta/spack-k4geo/spack/share/spack/setup-env.sh
spack load "k4geo@00-25 ^dd4hep+hepmc3"
export K4GEO=$(spack location -i "k4geo@00-25 ^dd4hep+hepmc3")/share/k4geo
export DET
GEOM=$RUN/geom_${DTAG}_${FIELD}.xml; STEER=$RUN/fulldet/SteeringFile_fulldet.py
OVERLAY=/pscratch/sd/s/sapta/bilevel/escan/prod/Z/bkg_Z_1x.hepmc
OUT=$RUN/sim_outputs/test_${DTAG}; mkdir -p $OUT
for f in $GEOM $STEER $OVERLAY; do [ -s $f ] || { echo "missing $f"; exit 1; }; done
cd $RUN
echo "== $DET: 2 background crossings (Z 1x)"
/usr/bin/time -f "   wall %e s, max memory %M kB" ddsim --steeringFile $STEER --compactFile $GEOM --outputFile $OUT/bkg.root \
    -N 2 --random.seed 7000 --inputFiles $OVERLAY > $OUT/bkg.log 2>&1
echo "   ddsim exit code $?"; grep -E "StartUp Time|Total Time" $OUT/bkg.log; grep -i -E "^.*(error|exception|fatal)" $OUT/bkg.log | grep -v "G4Exception-START\|G4Exception-END" | cut -c1-160 | sort | uniq -c | sort -rn | head -6
echo "== $DET: 5 photons of 1 GeV, 60-120 degrees"
/usr/bin/time -f "   wall %e s, max memory %M kB" ddsim --steeringFile $STEER --compactFile $GEOM --outputFile $OUT/gun.root \
    -N 5 --random.seed 2000 --enableGun --gun.particle gamma --gun.distribution uniform \
    --gun.momentumMin "1*GeV" --gun.momentumMax "1*GeV" --gun.thetaMin "1.047*rad" --gun.thetaMax "2.094*rad" > $OUT/gun.log 2>&1
echo "   ddsim exit code $?"; grep -E "StartUp Time|Total Time" $OUT/gun.log
echo "== hit collections and entries per event (background file / gun file)"
python - $OUT/bkg.root $OUT/gun.root <<'PY' 2>/dev/null
import sys, ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError
for p in sys.argv[1:]:
    f = ROOT.TFile.Open(p); t = f.Get("events")
    if not t: print("  no events tree in", p); continue
    names = sorted(b.GetName() for b in t.GetListOfBranches() if not b.GetName().startswith("_") and "Contribution" not in b.GetName())
    print(f"  {p.split('/')[-1]}: {t.GetEntries()} events")
    for n in names:
        if n in ("MCParticles", "EventHeader"): continue
        t.Draw(f"@{n}.size()>>h(1,0,1e9)", "", "goff"); h = ROOT.gDirectory.Get("h")
        m = h.GetMean() if h else -1
        if m > 0: print(f"     {n:36} {m:10.0f} hits per event")
PY
echo "next: in the analysis environment,  python fulldet/check_fulldet_edep.py sim_outputs/test_${DTAG} --nfiles 1   (needs the file named events_*.root: cp $OUT/bkg.root $OUT/events_000.root)"
