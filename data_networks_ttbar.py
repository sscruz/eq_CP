from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import tables 
import pandas as pd 
import torch 

class dataset( IterableDataset ):
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


            print(thedata.columns)
            control_vars=torch.Tensor( thedata[['control_cnr_crn','control_cnk_kn','control_rk_kr']].values).to(self.device)
            weights  =torch.Tensor(thedata[["weight_sm", "weight_lin",
                                             "weight_quad"]].values).to(self.device)
            variables=torch.Tensor(thedata[['lep_px', 'lep_py', 'lep_pz',						#único lepton
                                             'b1_px', 'b1_py', 'b1_pz',
                                             'b2_px', 'b2_py', 'b2_pz',
                                             'light1_px', 'light1_py', 'light1_pz', 			#quark1
                                             'light2_px', 'light2_py', 'light2_pz',				#quark2
                                             'met_px', 'met_py',
                                             'lep_charge']].values).to(self.device)
            yield from zip(weights, control_vars, variables)



class network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(18,80),
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

        cpx= torch.stack([-x[:,0], -x[:,1] , -x[:,2],    # -lep 
                          -x[:,6], -x[:,7] , -x[:,8],    # -b2 
                          -x[:,3], -x[:,4], -x[:,5],   # -b1
                          -x[:,12], -x[:,13] , -x[:,14],    # -q2 
                          -x[:,9],-x[:,10], -x[:,11],	#-q1	
                          -x[:,15],-x[:,16],			#-met
                           -x[:,17]],  					# -charge
                         dim=1).to(self.device)

        return self.main_module(x)-self.main_module(cpx)
