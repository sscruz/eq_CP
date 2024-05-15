from torch.utils.data import Dataset, Dataset
import torch.nn as nn
import glob 
import tables 
import pandas as pd 
import torch 

class dataset( Dataset ):
    def __init__( self, path, device):
        self.files=glob.glob(path)
        self.device=device
        self.length=0

        dfs=[]
        for f in self.files:
            thedata=pd.read_hdf(f, 'df')
            dfs.append(thedata)

        big_df=pd.concat( dfs )
        
        self.control_vars=torch.Tensor(big_df[["weight_sm", "weight_lin_ctZI"]].values ).to(self.device)
        self.weights  =torch.Tensor(big_df[[
            "weight_sm",
            "weight_lin_ctZI","weight_quad_ctZI",
            "weight_lin_ctZ","weight_quad_ctZ",
            #'weight_cross_ctZI_ctZ', 
        ]].values).to(self.device)
        self.variables=torch.Tensor(big_df[['photon_px', 'photon_py', 'photon_pz',                                            
                                            'lep_top_px', 'lep_top_py', 'lep_top_pz',
                                            'random_b1_px', 'random_b1_py', 'random_b1_pz',
                                            'random_b2_px', 'random_b2_py', 'random_b2_pz',
                                            'random_j1_px', 'random_j1_py', 'random_j1_pz',
                                            'random_j2_px', 'random_j2_py', 'random_j2_pz',
                                            'met_px', 'met_py',
                                            'lep_top_sign'
                                        ]].values).to(self.device)
        self.length=len(big_df)


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.weights[i,:], self.control_vars[i,:], self.variables[i,:]




class network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(21,80),
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

        cpx= torch.stack([-x[:,0], -x[:,1] , -x[:,2],    # -photon
                          -x[:,3], -x[:,4] , -x[:,5],    # -lep top 
                          -x[:,9],-x[:,10], -x[:,11],   # -random b2 # these shouldnt matter 
                          -x[:,6], -x[:,7], -x[:,8],   # -random b1 # these shouldnt matter
                          -x[:,15],-x[:,16], -x[:,17],   # -random j2 # these shouldnt matter
                          -x[:,12],-x[:,13], -x[:,14],   # -random j1 # these shouldnt matter
                          -x[:,18],-x[:,19],             # -met 
                          -x[:,20],                      # -top lepton charge
                      ],  
                         dim=1).to(self.device)

        return self.main_module(x)-self.main_module(cpx)
