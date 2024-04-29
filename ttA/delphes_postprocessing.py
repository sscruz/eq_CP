import sys
import random 

import ROOT
import multiprocessing 

from tqdm import tqdm
import pandas as pd 
from collections import defaultdict 

ROOT.gSystem.Load("libDelphes")

try:
  ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
  ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
except:
  pass


def process_file( inp ):
  fil, number = inp

  chain = ROOT.TChain("Delphes")
  chain.Add(fil)

  ret=defaultdict(list)

  # Create object of class ExRootTreeReader
  treeReader = ROOT.ExRootTreeReader(chain)
  numberOfEntries = treeReader.GetEntries()

  # Get pointers to branches used in this analysis
  branchWeight   = treeReader.UseBranch("Weight")
  branchEvent    = treeReader.UseBranch("Event")
  branchParticle = treeReader.UseBranch("Particle")
  branchPhoton   = treeReader.UseBranch("Photon")
  branchElectron = treeReader.UseBranch("Electron")
  branchMuon     = treeReader.UseBranch("Muon")
  branchJet      = treeReader.UseBranch("Jet")
  branchMET      = treeReader.UseBranch("MissingET")


  def get_tops( branchParticle ):
    
    selected=[]
    for p in branchParticle:
        if abs(p.PID) not in [6]: continue
        if p.Status != 22: continue
        selected.append(p)
    return selected

  def get_leps_from_top( branchParticle ):
  
    selected_leps=[]
    for i,p in enumerate(branchParticle):
      if abs(p.PID) not in [11,13]: continue
      isPrompt=False
      mother=p
      ascentry=[] 
      while mother.M1 >= 0:
        mother=branchParticle[mother.M1]
        ascentry.append( mother.PID ) 
        if abs(mother.PID) == 6:
          break
  
      
      for prompt in [6,-6,24,-24]:
        while prompt in ascentry:
          ascentry.remove(prompt)
  
      if len(ascentry) > 0: continue
      selected_leps.append( p)
  
    return selected_leps

  def get_photon( branchParticle ):
  
    selected=[]
    for p in branchParticle:
        if p.PID != 22: continue
        if p.Status != 23: continue
        selected.append(p)
    return selected


  def invert_matrix( sm, w1, w2, a, b):
    L=(b*b*w1-a*a*w2+a*a*sm-b*b*sm)/(a*b*b-b*a*a)
    Q=(b*w1-a*w2+(a-b)*sm)/(a*a*b-b*b*a)
    return L, Q

  def process_weights( branchWeight ):
    
    SM=branchWeight[5].Weight

    ctZI_1=branchWeight[6].Weight
    ctZI_m2=branchWeight[7].Weight

    lin_ctZI, quad_ctZI=invert_matrix(SM, ctZI_1, ctZI_m2, 1, -2)

    ctZ_3=branchWeight[8].Weight
    ctZ_m4=branchWeight[9].Weight

    lin_ctZ, quad_ctZ=invert_matrix(SM, ctZ_3, ctZ_m4, 3, -4)

    for_cross=branchWeight[10].Weight
    cross = for_cross-SM-lin_ctZI-quad_ctZI-lin_ctZ-quad_ctZ

    return SM, lin_ctZI, quad_ctZI, lin_ctZ, quad_ctZ, cross


  count=0
  # Loop over all events
  pbar=tqdm(range(0, numberOfEntries))
  for entry in pbar:
    # Load selected branches with data from specified event
    treeReader.ReadEntry(entry)


    # photon selection
    selected_photons=[]
    for ph in branchPhoton:
      if abs(ph.Eta) > 2.5 or ph.PT < 30: continue
      selected_photons.append(ph)

    if len(selected_photons) < 1: continue

    
    # # lepton selection
    selected_leptons=[]
    for lep in [x for x in branchMuon]+[x for x in branchElectron]:
      if abs(lep.Eta) > 2.5 or lep.PT < 25: continue
      selected_leptons.append(lep)

    if len(selected_leptons) < 1: continue
    count = count+1
    pbar.set_description(f"eff: {count/(entry+1)}")

 
    ## jet selection
    selected_jets=[]
    for j in branchJet:
      if abs(j.Eta) > 2.5 or j.PT < 30: continue
      is_clean=True
      vj=ROOT.TLorentzVector()
      vj.SetPtEtaPhiM( j.PT, j.Eta, j.Phi, 0)
      for l in selected_leptons:
        vl=ROOT.TLorentzVector()
        vl.SetPtEtaPhiM(l.PT, l.Eta, l.Phi, 0)
        if vl.DeltaR( vj ) < 0.4: 
          is_clean=False 
          break
      if is_clean:
        selected_jets.append( j ) 

    if len(selected_jets) < 4: continue



    bjets=[j for j in selected_jets if j.BTag > 0]
    if len(bjets) < 1: continue 


    b1=bjets[0]
    selected_jets.remove(b1)
    b2=bjets[1] if len(bjets) > 1 else selected_jets[0]
    selected_jets.remove(b2)

    j1=selected_jets[0]
    j2=selected_jets[1]

    bs=[b1,b2]
    random.shuffle( bs )
    b1,b2=bs

    js=[j1,j2]
    random.shuffle( js )
    j1,j2=js
    


    for label, part in [('lep_top', selected_leptons[0]), ('random_b1',b1), ('random_b2',b2), ('random_j1',j1), ('random_j2',j2), ('photon',selected_photons[0])]:
      vpart = ROOT.TLorentzVector()
      vpart.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, 0)
      for what in ['Px', 'Py','Pz']:
        ret[label+'_'+what.lower()].append( getattr(vpart, what)() ) 


    vmet=ROOT.TVector3()
    vmet.SetPtEtaPhi( branchMET[0].MET, 0, branchMET[0].Phi)
    ret['met_px'].append( vmet.Px() ) 
    ret['met_py'].append( vmet.Py() )
    ret['lep_top_sign'].append( selected_leptons[0].Charge ) 


    sm, lin_ctZI, quad_ctZI, lin_ctZ, quad_ctZ, cross_ctZI_ctZ = process_weights( branchWeight )
    ret['weight_sm' ].append( sm )
    ret['weight_lin_ctZI' ].append( lin_ctZI )
    ret['weight_quad_ctZI'].append(quad_ctZI )
    ret['weight_lin_ctZ'  ].append(lin_ctZ ) 
    ret['weight_quad_ctZ' ].append(quad_ctZ ) 
    ret['weight_cross_ctZI_ctZ' ].append(cross_ctZI_ctZ)

  
  df=pd.DataFrame.from_dict(ret)
  outf=fil.split("/")[-1].replace(".root", "%d.h5"%number)
  df.to_hdf( outf, 'df')

  return 

file_list=[
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_0/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_10/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_11/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_12/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_13/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_14/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_15/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_1/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_2/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_3/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_4/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_5/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_6/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_7/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_8/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat//store/user/sesanche/CPV/ttA_delphes_interf/chunk_9/run_01/tag_1_delphes_events.root",
]

from multiprocessing import Pool
pool=Pool(10)
pool.map( process_file,  [(x,i) for i,x in enumerate(file_list) ])


