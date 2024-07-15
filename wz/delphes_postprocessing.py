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
  branchElectron = treeReader.UseBranch("Electron")
  branchMuon     = treeReader.UseBranch("Muon")
  branchMET      = treeReader.UseBranch("MissingET")




  def invert_matrix( sm, w1, w2, a, b):
    L=(b*b*w1-a*a*w2+a*a*sm-b*b*sm)/(a*b*b-b*a*a)
    Q=(b*w1-a*w2+(a-b)*sm)/(a*a*b-b*b*a)
    return L, Q

  def process_weights( branchWeight ):
    
    SM      = branchWeight[2].Weight
    cWtil_dn = branchWeight[3].Weight
    cWtil_up = branchWeight[4].Weight

    lin_cWtil, quad_cWtil=invert_matrix(SM, cWtil_up, cWtil_dn, 1, -1)

    return SM, lin_cWtil, quad_cWtil

  def get_gen_things( branchParticle ):
    w=None; z=None
    for p in branchParticle:
      if p.Status != 22: continue
      if abs(p.PID) == 24: w=p
      if abs(p.PID) == 23: z=p
      if w is not None and z is not None: break
    if w is None: raise RuntimeError
    if z is None: raise RuntimeError

    w_decays=[]; z_decays=[]
    for p in branchParticle:
      if abs(p.PID) not in [11,13,12,14]: continue
      isPrompt=False
      mother=p
      ascentry=[] 
      while mother.M1 >= 0:
        if mother.PID == p.PID:
          p=mother
        mother=branchParticle[mother.M1]
        if mother == w:
          w_decays.append(p)
          break
        elif mother == z:
          z_decays.append(p)
          break

    def cleanup(parts):
      parts = list(set(parts))
      toremove=[]
      if any( map( lambda x : x.Status != 23, parts)) and len(parts) != 2:
        for x in parts:
          if x.Status != 23: 
            toremove.append(x)
      for x in toremove: parts.remove(x)
      return parts 

    w_decays=cleanup(w_decays)
    z_decays=cleanup(z_decays)




    if len(w_decays) != 2: return None, None, None, None
    if len(z_decays) != 2: return None, None, None, None

    return w, z, w_decays, z_decays

  def get_vec(part):
    vec=ROOT.TLorentzVector()
    vec.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, getattr(part, 'Mass', 0))
    return vec


  count=0; passed=0
  # Loop over all events
  pbar=tqdm(range(0, numberOfEntries))

  for entry in pbar:
    # Load selected branches with data from specified event
    treeReader.ReadEntry(entry)

    count = count + 1 
    


    # # lepton selection
    selected_leptons=[]
    muons     = [x for x in branchMuon]   ; 
    for x in muons:
      setattr(x, 'PID', 13*x.Charge)
    electrons = [x for x in branchElectron]; 
    for x in electrons: 
      setattr(x, 'PID', 11*x.Charge)

    for lep in muons+electrons:
      if abs(lep.Eta) > 2.5 or lep.PT < 15: continue
      selected_leptons.append(lep)

    if len(selected_leptons) < 3: continue
    if selected_leptons[0].PT < 25: continue


    selected_leptons = selected_leptons[:3]
    ilepz_p=-1; ilepz_m=-1; ilepw=-1; mZ=-999999
    for i1,l1 in enumerate(selected_leptons):
      for i2,l2 in enumerate(selected_leptons):
        if i2>=i1: continue
        if l1.PID != -l2.PID: continue
        v1=get_vec(l1)
        v2=get_vec(l2)
        mass=(v1+v2).M()
        if abs(mass-91) < abs(mZ-91):
          mZ=mass
          ilepz_p=i1 if l1.PID > 0 else i2
          ilepz_m=i2 if l1.PID > 0 else i1




    if ilepz_p < 0: continue

    leps=[0,1,2]; leps.remove(ilepz_p); leps.remove(ilepz_m)
    lepw=selected_leptons[leps[0]]
    lepz_p=selected_leptons[ilepz_p]
    lepz_m=selected_leptons[ilepz_m]



    # now lets get the control variables
    w,z,w_decays,z_decays=get_gen_things( branchParticle )    
    if w is None: continue

    vz=get_vec(z); vw=get_vec(w)
    wz=vz+vw

    boost_wz = wz.BoostVector()
    boost_z  = vz.BoostVector()
    boost_w  = vw.BoostVector()
    
    w_rf = deepcopy(vw)
    w_rf.Boost( -boost_wz )
    
    z_rf = deepcopy(vz)
    z_rf.Boost( -boost_wz )

    rhat = -(wz.Vect().Unit())
    zhat=deepcopy( w_rf ).Vect().Unit()
    yhat=(zhat.Cross(rhat)).Unit()
    xhat=(yhat.Cross(zhat)).Unit()

    zplus = z_decays[0] if z_decays[0].PID > 0 else z_decays[1]
    zlm_z_restframe=get_vec(zplus); zlm_z_restframe.Boost( -boost_z)
    zlm_z_restframe=zlm_z_restframe.Vect()
    ret['phi_z'].append( ROOT.TVector3( zlm_z_restframe.Dot(xhat), zlm_z_restframe.Dot(yhat), zlm_z_restframe.Dot(zhat)).Phi())

    # w decay products in the w frame
    wlep = w_decays[0] if abs(w_decays[0].PID) in [11,13] else w_decays[1] 
    wl_w_restframe = get_vec( wlep ); wl_w_restframe.Boost( -boost_w)
    wl_w_restframe=wl_w_restframe.Vect()
    ret['phi_w'].append( ROOT.TVector3(wl_w_restframe.Dot(xhat), wl_w_restframe.Dot(yhat), wl_w_restframe.Dot(zhat)).Phi() ) 


    passed=passed+1
    pbar.set_description(f'Eff: {passed/count}')

    # no continues after here 
    
    for label, part in [('lzp', lepz_p), ('lzm', lepz_m), ('lw', lepw)]:
      vpart = ROOT.TLorentzVector()
      vpart.SetPtEtaPhiM( part.PT, part.Eta, part.Phi, 0)
      for what in ['Px', 'Py','Pz']:
        ret[label+'_'+what.lower()].append( getattr(vpart, what)() ) 

    ret['lwsign'].append( lepw.PID/abs(lepw.PID) ) 

    vmet=ROOT.TVector3()
    vmet.SetPtEtaPhi( branchMET[0].MET, 0, branchMET[0].Phi)
    ret['met_px'].append( vmet.Px() ) 
    ret['met_py'].append( vmet.Py() )

    sm, lin_cwtil, quad_cwtil = process_weights( branchWeight )
    ret['weight_sm' ]        .append( sm )
    ret['weight_cwtil_sm' ]  .append( lin_cwtil )
    ret['weight_cwtil_cwtil'].append( quad_cwtil )

  
  df=pd.DataFrame.from_dict(ret)
  outf=fil.split("/")[-1].replace(".root", "%d.h5"%number)
  df.to_hdf( outf, 'df')

  return 

file_list=[
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_0/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_1/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_2/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_3/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_4/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_5/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_6/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_7/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_8/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_9/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_10/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_11/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_12/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_13/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_14/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_15/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_16/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_17/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_18/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_19/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_20/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_21/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_22/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_23/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_24/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_25/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_26/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_27/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_28/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_29/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_30/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_31/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_32/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_33/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_34/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_35/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_36/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_37/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_38/run_01/tag_1_delphes_events.root",
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_39/run_01/tag_1_delphes_events.root",
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v2/chunk_40/run_01/tag_1_delphes_events.root",
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_01/tag_2_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_02/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_03/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_04/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_05/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_06/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_07/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_08/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_09/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_10/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_11/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_12/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_13/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_14/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_15/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_16/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_17/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_18/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_19/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_20/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes/part2/run_21/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_0/run_02/tag_2_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_10/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_11/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_12/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_13/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_14/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_15/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_16/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_17/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_18/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_19/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_1/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_20/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_21/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_22/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_23/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_24/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_25/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_26/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_27/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_28/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_29/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_2/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_30/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_31/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_32/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_33/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_34/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_35/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_36/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_37/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_38/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_39/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_3/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_40/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_41/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_42/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_43/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_44/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_45/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_46/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_47/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_48/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_49/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_4/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_50/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_51/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_52/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_53/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_54/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_55/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_56/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_57/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_58/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_59/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_5/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_60/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_61/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_62/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_63/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_64/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_65/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_66/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_67/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_68/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_69/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_6/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_70/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_71/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_72/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_73/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_74/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_75/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_76/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_7/run_01/tag_1_delphes_events.root", 
 "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_8/run_01/tag_1_delphes_events.root", 
"/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CPV/wz_delphes_v3/chunk_9/run_01/tag_1_delphes_events.root", 

]

from multiprocessing import Pool
pool=Pool(10)
pool.map( process_file,  [(x,i) for i,x in enumerate(file_list) ])


