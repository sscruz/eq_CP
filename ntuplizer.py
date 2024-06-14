import ROOT as r 
import pandas as pd 
from tqdm import tqdm 
from lhereader import LHEReader
import random
import math 
from copy import deepcopy
import numpy as np 
import glob 
from multiprocessing import Pool 

def process_file( fil ):
    ret=[]
    reader = LHEReader(fil,weight_mode='dict')
    for event in reader:

        # first the weights
        w=event.weights
        sm    = w['rw0000']
        op_dn = w['rw0001']
        op_up = w['rw0002']

        lin=(op_up-op_dn)/2
        quad=(op_up+op_dn-2*sm)/2
        toret=[sm, lin, quad]

        # now the particles
        part_list=[('lp',11), ('lm',-11), ('bp',5), ('bm',-5), ('nup',12), ('num',-12), ('tp',6), ('tm',-6),('dp',1),('dm',-1),('up',2),('um',-2),('sp',3),('sm',-3),('cp',4),('cm',-4)]
        particles={}
        for label, pdgid in part_list:
            particles[label]=r.TLorentzVector()
            for p in event.particles:
                if p.status <0: 
                    continue
                if p.pdgid == pdgid:
                    particles[label].SetPxPyPzE( p.px, p.py, p.pz, p.energy) 
                    break
        
        bs=[particles['bp'],particles['bm']]
        random.shuffle( bs ) 
        
        lights=[]
        for label in ['dp','dm','up','um','sp','sm','cp','cm']:
            if particles[label].E()!=0:
                lights.append(particles[label])
        
        if len(lights)!=2:
            print("alarm")
        
        random.shuffle(lights)
		
        lep=[particles['lp'] if particles['lm'].E()==0 else particles['lm']]
        charge=[1 if lep[0]==particles['lm'] else -1]
        input_particles = [lep[0], bs[0], bs[1], lights[0],lights[1]]

        for p in input_particles:
            for what in "Px,Py,Pz".split(","):
                toret.append( getattr(p,what)())

        nus = particles['nup']+particles['num']
        toret.extend([ nus.Px(), nus.Py(), charge])

        # now high-level (control) variables

        # lets build the rest frame 
        ttbar = particles['tp']+particles['tm']
        boost_ttbar = ttbar.BoostVector()
        
        tp_rest_frame=deepcopy(particles['tp'])
        tp_rest_frame.Boost( -boost_ttbar )

        k_hat=tp_rest_frame.Vect().Unit()
        p_hat = r.TVector3(0,0,1)
        y = k_hat.Dot(p_hat)
        sign_ =  float(np.sign(y)) # Bernreuther Table 5        
        rval = math.sqrt( 1-y**2 ) 

        r_hat = sign_/rval*(p_hat - (k_hat*y) )
        n_hat = sign_/rval*(p_hat.Cross(k_hat))


        # phat=tp_rest_frame.Vect().Unit()

        # def print_vector(v):
        #     print(v.Px(),v.Py(),v.Pz())

        # for incoming_particle in event.particles:
        #     if incoming_particle.status < 0:
        #         v_parton_direction=r.TLorentzVector()
        #         v_parton_direction.SetPxPyPzE( incoming_particle.px, incoming_particle.py, incoming_particle.pz, incoming_particle.energy)    
        #         v_parton_direction.Boost( -boost_ttbar )
        #         khat=v_parton_direction.Vect().Unit()
        #         #print_vector(khat)
        #         break

        # y=phat.Dot(khat)
        # #sign=float(np.sign(y))
        # rval=math.sqrt(1-y*y)

        # rhat=(phat-y*khat)        #(sign/rval)*
        # nhat=(phat.Cross(khat))   #(sign/rval)*

        # now profit from it 

        #lep=[deepcopy(particles['lp']) if lep[0]==particles['lp'] else deepcopy(particles['lm'])]
        #lep=lep[0]
		
        if lep[0]==particles['lp']:
            lep=deepcopy(particles['lp'])
            for label in ['dp','dm','sp','sm']:
                if particles[label]==lights[0]:
                    lig=deepcopy(particles[label])
                if particles[label]==lights[1]:
                    lig=deepcopy(particles[label])
        else:
            lig=deepcopy(particles['lm'])
            for label in ['dp','dm','sp','sm']:
                if particles[label]==lights[0]:
                    lep=deepcopy(particles[label])
                if particles[label]==lights[1]:
                    lep=deepcopy(particles[label])

	
        #for label in ['dp','dm','up','um','sp','sm','cp','cm']:
         #   if particles[label]==lights[0]:
          #      lights1=deepcopy(particles[label])
           # if particles[label]==lights[1]:
            #    lights2=deepcopy(particles[label])
		
        #lig=lights1+lights2
		
        lep.Boost(-boost_ttbar)
        lig.Boost(-boost_ttbar)

        lep_hat = lep.Vect().Unit()
        lig_hat = lig.Vect().Unit()


        #print(nhat.Dot(lphat)*rhat.Dot(lmhat), rhat.Dot(lphat)*nhat.Dot(lmhat))
        #print(kk)
        cnr_crn = n_hat.Dot(lep_hat)*r_hat.Dot(lig_hat)-r_hat.Dot(lep_hat)*n_hat.Dot(lig_hat)
        cnk_ckn = n_hat.Dot(lep_hat)*k_hat.Dot(lig_hat)-k_hat.Dot(lep_hat)*n_hat.Dot(lig_hat)
        crk_ckr = r_hat.Dot(lep_hat)*k_hat.Dot(lig_hat)-k_hat.Dot(lep_hat)*r_hat.Dot(lig_hat)
        
        toret.extend([cnr_crn,cnk_ckn, crk_ckr])


            

        ret.append(toret)


    cols=['weight_sm','weight_lin','weight_quad']+ ['%s_%s'%(part, what) for part in 'lep,b1,b2,light1,light2'.split(",") for what in 'px,py,pz'.split(",") ]+['met_px','met_py', 'lep_charge']+['control_cnr_crn','control_cnk_kn','control_rk_kr']
    df=pd.DataFrame( ret, columns=cols)    
    df.to_hdf(fil.replace("unweighted_events_","ntuple_").replace('.lhe','.h5'),'df')
    df.replace("/lustrefs/hdd_pool_dir/eq_ntuples/ttbar_semi_decomp/", "")

    #return ret
import os

username = os.environ.get('USERNAME')

if username == 'uo278174':
    outputpath = "/nfs/fanae/user/uo278174/[TFG]/eq_CP"
else:
    outputpath = "/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CP_equivariant"
files = glob.glob(f"/lustrefs/hdd_pool_dir/eq_ntuples/ttbar_semi_decomp/*.lhe")
#print(files)
pool=Pool(15)
pool.map( process_file, files)

