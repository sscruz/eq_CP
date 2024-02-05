import ROOT as r 
import pandas as pd 
from tqdm import tqdm 
from lhereader import LHEReader
import random
import math 
from copy import deepcopy
import numpy as np 
from glob import glob 
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
        part_list=[('lp',11), ('lm',-11), ('bp',5), ('bm',-5), ('nup',12), ('num',-12), ('tp',6), ('tm',-6)]
        particles={}
        for label, pdgid in part_list:
            particles[label]=r.TLorentzVector()
            for p in event.particles:
                if p.pdgid == pdgid:
                    particles[label].SetPxPyPzE( p.px, p.py, p.pz, p.energy) 
                    break
        
        bs=[particles['bp'],particles['bm']]
        random.shuffle( bs ) 
        
        
        input_particles = [particles['lp'], particles['lm'], bs[0], bs[1]]

        for p in input_particles:
            for what in "Px,Py,Pz".split(","):
                toret.append( getattr(p,what)())

        nus = particles['nup']+particles['num']
        toret.extend([ nus.Px(), nus.Py()])

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
        lp=deepcopy(particles['lp'])
        lm=deepcopy(particles['lm'])

        lp.Boost(-boost_ttbar)
        lm.Boost(-boost_ttbar)

        lp_hat = lp.Vect().Unit()
        lm_hat = lm.Vect().Unit()

        #print(nhat.Dot(lphat)*rhat.Dot(lmhat), rhat.Dot(lphat)*nhat.Dot(lmhat))
        #print(kk)
        cnr_crn = n_hat.Dot(lp_hat)*r_hat.Dot(lm_hat)-r_hat.Dot(lp_hat)*n_hat.Dot(lm_hat)
        cnk_ckn = n_hat.Dot(lp_hat)*k_hat.Dot(lm_hat)-k_hat.Dot(lp_hat)*n_hat.Dot(lm_hat)
        crk_ckr = r_hat.Dot(lp_hat)*k_hat.Dot(lm_hat)-k_hat.Dot(lp_hat)*r_hat.Dot(lm_hat)
        
        toret.extend([cnr_crn,cnk_ckn, crk_ckr])


            

        ret.append(toret)


    cols=['weight_sm','weight_lin','weight_quad']+ ['%s_%s'%(part, what) for part in 'lp,lm,b1,b2'.split(",") for what in 'px,py,pz'.split(",") ]+['met_px','met_py']+['control_cnr_crn','control_cnk_kn','control_rk_kr']
    df=pd.DataFrame( ret, columns=cols)    
    df.to_hdf(fil.replace("unweighted_events_","ntuple_").replace('.lhe','.h5'),'df')
    #return ret

files=glob("/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CP_equivariant/ttbar/*.lhe")
pool=Pool(15)
pool.map( process_file, files)

