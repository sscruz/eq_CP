from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import tables 
import pandas as pd 
import torch 

class data_ttbar( IterableDataset ):
    def __init__( self, path, device):
        self.files=glob.glob(path)
        self.device=device
        self.length=0
        for fil in self.files:
            h5file=tables.open_file(fil, mode='r')
            self.length+=h5file.root.df.axis1.shape[0]
            h5file.close()

    def __len__(self):
        return self.length

    def __iter__(self):
        for f in self.files:
            thedata=pd.read_hdf(f, 'df')
            control_vars=torch.Tensor( thedata[['control_cnr_crn','control_cnk_kn','control_rk_kr']].values).to(self.device)
            weights  =torch.Tensor(thedata[["weight_sm", "weight_lin",
                                             "weight_quad"]].values).to(self.device)
            variables=torch.Tensor(thedata[['lp_px', 'lp_py', 'lp_pz',
                                             'lm_px', 'lm_py', 'lm_pz',
                                             'b1_px', 'b1_py', 'b1_pz',
                                             'b2_px', 'b2_py', 'b2_pz',
                                             'met_px', 'met_py']].values).to(self.device)
            yield from zip(weights, control_vars, variables)



class ttbar_net(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            #nn.Linear(14,80),
            nn.Linear(1,80),
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

        # sergiocpx= torch.stack([-x[:,3], -x[:,4] , -x[:,5],    # -lep minus
        # sergio                  -x[:,0], -x[:,1] , -x[:,2],    # -lep plus
        # sergio                  -x[:,9], -x[:,10], -x[:,11],   # -b2 # this shouldnt add information but ok 
        # sergio                  -x[:,6], -x[:,7] , -x[:,8],    # -b1 
        # sergio                  -x[:,12],-x[:,13]          ],  # -met
        # sergio                 dim=1)

        cpx = -x # sergio
        return self.main_module(x)-self.main_module(cpx)
