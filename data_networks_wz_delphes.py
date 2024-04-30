# python train.py --name wz_recopz_feb27 --data-path minitrees/wz_training_withpz --analysis wz --data-format root --batch-size 1000 --lr 1e-4
from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import torch 
import uproot 
import tables
import pandas as pd 
import numpy as np 

class dataset( IterableDataset ):
    def __init__( self, path, device):
        self.files=glob.glob(path)
        self.device=device
        self.length=0
        self.var_range = [[ -np.pi, np.pi], [-np.pi, np.pi]]
        dfs=[]
        for fil in self.files:
            h5file=tables.open_file(fil, mode='r')
            self.length+=h5file.root.df.axis1.shape[0]
            h5file.close()  

    def __len__(self):
        return self.length

    def __iter__(self):
        for fil in self.files:
            thedata=pd.read_hdf(fil, 'df')

            control_vars=torch.Tensor( thedata[['phi_z','phi_w']].values ).to(self.device)
            weights     =torch.Tensor( thedata[["weight_sm", "weight_cwtil_sm"]].values).to(self.device)
            variables   =torch.Tensor( thedata[['lzp_px', 'lzp_py', 'lzp_pz',
                                                'lzm_px', 'lzm_py', 'lzm_pz',
                                                'lw_px', 'lw_py', 'lw_pz',
                                                'met_px', 'met_py',
                                                'lwsign']].values).to(self.device)
            yield from zip(weights, control_vars, variables)

class network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(12,80),
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
        cpx= torch.stack([-x[:,3], -x[:,4] , -x[:,5],           # -lep z minus 3-momenta
                          -x[:,0], -x[:,1] , -x[:,2],           # -lep z plus  3-momenta
                          -x[:,6], -x[:,7] , -x[:,8],           # -W 3-momenta
                          -x[:,9], -x[:,10],                    # -met
                          -x[:,11]],                            # -W sign
                         dim=1)
        return self.main_module(x)-self.main_module(cpx)
