# python train.py --name ttbar_particle_level --data-path /work/sesanche/ttH/UL/CMSSW_10_4_0/src/CMGTools/TTHAnalysis/macros/kk     --analysis ttbar_pl

from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import torch 
import uproot 
import tables 
import pandas as pd 
class dataset( Dataset ):
    def __init__( self, path,  device):
        self.files=glob.glob(path)
        self.device=device

        dfs=[]
        for f in self.files:
            thedata=pd.read_hdf(f, 'df')
            dfs.append(thedata)
        big_df=pd.concat( dfs )
        self.length=len(big_df)


        self.control_vars=torch.Tensor( big_df[['parton_cnr_crn','parton_cnk_ckn','parton_crk_ckr']].values).to(self.device)
        self.weights  =torch.Tensor(big_df[["weight_sm", "weight_lin_ctGI"]].values).to(self.device)
        self.variables=torch.Tensor(big_df[['lep_pos_px', 'lep_pos_py', 'lep_pos_pz',
                                             'lep_neg_px', 'lep_neg_py', 'lep_neg_pz',
                                             'random_b1_px', 'random_b1_py', 'random_b1_pz',
                                             'random_b2_px', 'random_b2_py', 'random_b2_pz',
                                             'met_px', 'met_py']].values).to(self.device)


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.weights[i,:], self.control_vars[i,:], self.variables[i,:]
    
    def name_cvaris(self,index):
        if index == 0:
            return  "$c_{rn} - c_{nr}$"
        elif index == 1:
            return  "$c_{kn} - c_{nk}$"
        elif index == 2:
            return  "$c_{rk} - c_{kr}$"
               
class network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(14,80),
            nn.LeakyReLU(),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1 ),
        )
        self.main_module.to(device)
        self.device=device

    def forward(self, x):

        cpx= torch.stack([-x[:,3], -x[:,4] , -x[:,5],    # -lep minus
                          -x[:,0], -x[:,1] , -x[:,2],    # -lep plus
                          -x[:,9], -x[:,10], -x[:,11],   # -b2 # this shouldnt add information but ok 
                          -x[:,6], -x[:,7] , -x[:,8],    # -b1 
                          -x[:,12],-x[:,13]          ],  # -met
                         dim=1).to(self.device)

        return self.main_module(x)-self.main_module(cpx)
        
class network_noeq(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(14,80),
            nn.LeakyReLU(),
            nn.Linear(80, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1 ),
        )
        self.main_module.to(device)

    def forward(self, x):
        return self.main_module(x)
