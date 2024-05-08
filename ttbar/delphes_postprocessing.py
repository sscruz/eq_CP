import sys
import random 

import ROOT
import multiprocessing 

from tqdm import tqdm
import pandas as pd 
from collections import defaultdict 
from copy import deepcopy
import numpy as np 
import math 

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
    
    SM      = branchWeight[3].Weight
    ctGI_dn = branchWeight[4].Weight
    ctGI_up = branchWeight[5].Weight

    lin_ctGI, quad_ctGI=invert_matrix(SM, ctGI_up, ctGI_dn, 1, -1)

    return SM, lin_ctGI, quad_ctGI


  count=0; passed=0
  # Loop over all events
  pbar=tqdm(range(0, numberOfEntries))
  for entry in pbar:
    # Load selected branches with data from specified event
    treeReader.ReadEntry(entry)

    count = count + 1 
    


    # # lepton selection
    selected_leptons=[]
    for lep in [x for x in branchMuon]+[x for x in branchElectron]:
      if abs(lep.Eta) > 2.5 or lep.PT < 15: continue
      selected_leptons.append(lep)

    if len(selected_leptons) < 2: continue
    if selected_leptons[0].PT < 25: continue

 
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

    if len(selected_jets) < 2: continue
    lep_pos = selected_leptons[0] if selected_leptons[0].Charge > 0 else selected_leptons[1]
    lep_neg = selected_leptons[1] if selected_leptons[0].Charge > 0 else selected_leptons[0]


    bjets=[j for j in selected_jets if j.BTag > 0]
    if len(bjets) < 1: continue 

    passed=passed+1
    pbar.set_description(f'Eff: {passed/count}')

    b1=bjets[0]
    selected_jets.remove(b1)
    b2=bjets[1] if len(bjets) > 1 else selected_jets[0]
    selected_jets.remove(b2)


    bs=[b1,b2]
    random.shuffle( bs )
    b1,b2=bs


    # no continues after here 

    for label, part in [('lep_pos', lep_pos), ('lep_neg', lep_neg), ('random_b1',b1), ('random_b2',b2)]:
      vpart = ROOT.TLorentzVector()
      vpart.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, 0)
      for what in ['Px', 'Py','Pz']:
        ret[label+'_'+what.lower()].append( getattr(vpart, what)() ) 


    vmet=ROOT.TVector3()
    vmet.SetPtEtaPhi( branchMET[0].MET, 0, branchMET[0].Phi)
    ret['met_px'].append( vmet.Px() ) 
    ret['met_py'].append( vmet.Py() )

    sm, lin_ctGI, quad_ctGI = process_weights( branchWeight )
    ret['weight_sm' ].append( sm )
    ret['weight_lin_ctGI' ].append( lin_ctGI )
    ret['weight_quad_ctGI'].append( quad_ctGI )



    # now lets get the control variables
    tops=get_tops( branchParticle )
    assert( len(tops) == 2)
    top = ROOT.TLorentzVector(); atop = ROOT.TLorentzVector();
    for part in tops:
      if part.PID > 0: 
        top.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, part.Mass)
      else:
        atop.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, part.Mass)

    ttbar = top+atop
    boost_ttbar = ttbar.BoostVector()
    tp_rest_frame=deepcopy( top ) 
    tp_rest_frame.Boost( -boost_ttbar )

    k_hat=tp_rest_frame.Vect().Unit()
    p_hat = ROOT.TVector3(0,0,1)
    y = k_hat.Dot(p_hat)
    sign_ =  float(np.sign(y)) # Bernreuther Table 5        
    rval = math.sqrt( 1-y**2 ) 
    
    r_hat = sign_/rval*(p_hat - (k_hat*y) )
    n_hat = sign_/rval*(p_hat.Cross(k_hat))

    # now profit from it 
    leps_from_top=get_leps_from_top( branchParticle )
    assert( len(leps_from_top) == 2 )
    lp=ROOT.TLorentzVector(); lm=ROOT.TLorentzVector()
    for l in leps_from_top:
      if l.PID > 0:
        lp.SetPtEtaPhiM( l.PT, l.Eta, l.Phi, 0)
      else:
        lm.SetPtEtaPhiM( l.PT, l.Eta, l.Phi, 0)

    lp.Boost(-boost_ttbar)
    lm.Boost(-boost_ttbar)

    lp_hat = lp.Vect().Unit()
    lm_hat = lm.Vect().Unit()

    ret['parton_cnr_crn'].append( n_hat.Dot(lp_hat)*r_hat.Dot(lm_hat)-r_hat.Dot(lp_hat)*n_hat.Dot(lm_hat))
    ret['parton_cnk_ckn'].append( n_hat.Dot(lp_hat)*k_hat.Dot(lm_hat)-k_hat.Dot(lp_hat)*n_hat.Dot(lm_hat))
    ret['parton_crk_ckr'].append( r_hat.Dot(lp_hat)*k_hat.Dot(lm_hat)-k_hat.Dot(lp_hat)*r_hat.Dot(lm_hat))

  
  df=pd.DataFrame.from_dict(ret)
  outf=fil.split("/")[-1].replace(".root", "%d.h5"%number)
  df.to_hdf( outf, 'df')

  return 

file_list=[
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_02/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_03/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_04/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_05/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_06/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_07/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_08/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_09/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_10/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_11/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_12/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_13/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_14/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_15/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_16/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_17/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_18/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_19/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_20/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms/run_21/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_0/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_10/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_11/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_12/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_13/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_14/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_15/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_16/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_17/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_18/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_19/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_1/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_20/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_21/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_22/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_23/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_24/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_25/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_26/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_27/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_28/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_29/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_2/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_30/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_31/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_32/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_33/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_34/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_35/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_36/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_37/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_38/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_39/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_3/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_40/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_4/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_5/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_6/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_7/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_8/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_emu_delphes/chunk_9/run_01/tag_1_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_01/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_02/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_03/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_04/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_05/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_06/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_07/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_08/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_09/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_10/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_11/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_12/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_13/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_14/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_15/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_16/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_17/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_18/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_19/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_20/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_2/run_21/tag_2_delphes_events.root",
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_01/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_02/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_03/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_04/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_05/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_06/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_07/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_08/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_09/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_10/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_11/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_12/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_13/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_3/run_14/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_01/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_02/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_03/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_04/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_05/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_06/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_07/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_08/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_09/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_10/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_11/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_12/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_13/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_14/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_15/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_16/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_17/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_18/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_19/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_20/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_4/run_21/tag_2_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_01/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_02/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_03/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_04/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_05/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_06/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_07/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_08/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_09/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_10/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_11/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_12/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_13/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_14/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_15/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_16/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_17/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_18/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_19/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_20/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_21/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_22/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_23/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_24/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_25/tag_1_delphes_events.root", 
#   "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_5/run_26/tag_1_delphes_events.root",   
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_0/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_1/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_2/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_3/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_4/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_5/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_6/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_7/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_8/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_9/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_10/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_11/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_12/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_13/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_14/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_15/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_16/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_17/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_18/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_19/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_20/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_21/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_22/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/may2/chunk_23/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_02/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_03/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_04/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_05/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_06/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_07/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_08/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_09/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_10/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_11/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_12/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_13/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_14/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_15/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_16/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_17/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_18/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_19/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_20/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/ttbar_delphes/fromvoms_6/run_21/tag_1_delphes_events.root",
]

from multiprocessing import Pool
pool=Pool(10)
pool.map( process_file,  [(x,i+207) for i,x in enumerate(file_list) ])


