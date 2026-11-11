"""ddsim steering for the full-detector energy-scan runs (CLD_o2_v08, ALLEGRO_o1_v03, full IDEA_o2_v01).

Same physics and particle-handling settings as SteeringFile_SCEPCal_bkgonly.py, with the
detector-specific sensitive-detector mappings chosen from the environment variable DET.
Used for both job types:
  background only  (the sbatch passes --inputFiles <hepmc>; the gun stays off)
  signal gun       (the sbatch passes --enableGun and the gun parameters)

The WarpX pairs are already in the laboratory frame with the 30 mrad crossing, so
crossingAngleBoost MUST stay 0 (the official CLD/ALLEGRO steering files set 0.015 because
their generator events are head-on).
"""
import os
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import mm, MeV

DET = os.environ.get("DET", "")

SIM = DD4hepSimulation()
SIM.crossingAngleBoost = 0.0
SIM.enableDetailedShowerMode = False
SIM.enableG4GPS = False
SIM.enableG4Gun = False
SIM.enableGun = False                       # the gun jobs turn it on from the command line
SIM.inputFiles = []
SIM.physicsList = "FTFP_BERT"
SIM.printLevel = 3
SIM.runType = "batch"
SIM.vertexOffset = [0.0, 0.0, 0.0, 0.0]
SIM.vertexSigma = [0.0, 0.0, 0.0, 0.0]

# -- sensitive detectors: calorimeters keep every deposit (no energy filter), trackers 1 keV
SIM.action.calo = "Geant4ScintillatorCalorimeterAction"
SIM.action.calorimeterSDTypes = ["calorimeter"]
SIM.action.tracker = ("Geant4TrackerWeightedAction", {"HitPositionCombination": 2, "CollectSingleDeposits": False})
SIM.action.trackerSDTypes = ["tracker"]
SIM.action.mapActions = {}
if DET == "IDEA":
    # as in k4geo example/SteeringFile_IDEA_o2_v01.py
    SIM.action.mapActions["SCEPCal_MainLayer"] = "SCEPCal_MainSDAction"
    SIM.action.mapActions["SCEPCal_TimingLayer"] = "SCEPCal_TimingSDAction"
# IDEA_NODR=1: geometry without the dual-readout tubes (cross-check geometries, or the crystal-only
# standalone file); their sensitive-detector set-up is then skipped, since it would look for detectors
# that are not there
if DET == "IDEA" and os.environ.get("IDEA_NODR", "") != "1":
    SIM.action.mapActions["DREndcapTubes"] = "DRTubesSDAction"
    SIM.action.mapActions["DRBarrelTubes"] = "DRTubesSDAction"
    try:      # one sensitive detector per tube family instead of one per tube (memory, start-up time)
        SIM.geometry.regexSensitiveDetector["DREndcapTubes"] = {"Match": ["DRETS"], "OutputLevel": 4}
        SIM.geometry.regexSensitiveDetector["DRBarrelTubes"] = {"Match": ["DRBT"], "OutputLevel": 4}
    except AttributeError:
        print("SteeringFile_fulldet: this ddsim has no geometry.regexSensitiveDetector; the dual-readout tubes "
              "will be slow to set up")
if DET == "ALLEGRO":
    SIM.action.mapActions["DCH_v2"] = "Geant4TrackerAction"   # as in the official ALLEGRO steering
SIM.filter.filters = {
    "geantino": {"name": "GeantinoRejectFilter/GeantinoRejector", "parameter": {}},
    "edep1kev": {"name": "EnergyDepositMinimumCut", "parameter": {"Cut": 0.001}},
    "edep0": {"name": "EnergyDepositMinimumCut/Cut0", "parameter": {"Cut": 0.0}},
}
SIM.filter.calo = ""
SIM.filter.tracker = "edep1kev"
SIM.filter.mapDetFilter = {}
if DET == "ALLEGRO":
    SIM.filter.mapDetFilter["DCH_v2"] = "edep0"
if DET == "IDEA":
    SIM.filter.mapDetFilter["SCEPCal_MainLayer"] = ""
    SIM.filter.mapDetFilter["SCEPCal_TimingLayer"] = ""

# -- field stepping (same as the SCEPCal runs)
SIM.field.delta_chord = 0.25
SIM.field.delta_intersection = 0.001
SIM.field.delta_one_step = 0.01
SIM.field.eps_max = 0.001
SIM.field.eps_min = 5e-05
SIM.field.equation = "Mag_UsualEqRhs"
SIM.field.largest_step = 10000.0
SIM.field.min_chord_step = 0.01
SIM.field.stepper = "ClassicalRK4"

# -- particle handling (same as the SCEPCal runs)
SIM.part.enableDetailedHitsAndParticleInfo = False
SIM.part.keepAllParticles = False
SIM.part.minDistToParentVertex = 2.2e-14
SIM.part.minimalKineticEnergy = 1.0 * MeV
SIM.part.saveProcesses = ["Decay"]
SIM.part.userParticleHandler = "Geant4TCUserParticleHandler"
SIM.physics.decays = False
SIM.physics.list = "FTFP_BERT"
SIM.physics.rangecut = 0.7 * mm

SIM.hepmc3.useHepMC3 = True
SIM.output.geometry = 2
SIM.output.inputStage = 3
SIM.output.kernel = 3
SIM.output.part = 3
SIM.output.random = 6

# -- optical photons (IDEA only). The official IDEA_o2_v01 steering and the standalone SCEPCal runs
# switch on Cherenkov-photon generation, which fills the photon-count collections (..Ccounts) of
# the crystals and the dual-readout tubes. The deposited-energy collections used here
# (SCEPCal_MainEdep, SCEPCal_TimingEdep) are G4Step::GetTotalEnergyDeposit and do not depend on
# it, so it is OFF by default to save time; CHERENKOV=1 turns it on.
if DET == "IDEA" and os.environ.get("CHERENKOV", "") == "1":
    def setupCerenkov(kernel):
        from DDG4 import PhysicsList
        seq = kernel.physicsList()
        cerenkov = PhysicsList(kernel, "Geant4CerenkovPhysics/CerenkovPhys")
        cerenkov.MaxNumPhotonsPerStep = 1000
        cerenkov.VerboseLevel = 0
        cerenkov.enableUI()
        seq.adopt(cerenkov)
        ph = PhysicsList(kernel, "Geant4OpticalPhotonPhysics/OpticalGammaPhys")
        ph.addParticleConstructor("G4OpticalPhoton")
        ph.VerboseLevel = 0
        ph.enableUI()
        seq.adopt(ph)
        return None
    SIM.physics.setupUserPhysics(setupCerenkov)
