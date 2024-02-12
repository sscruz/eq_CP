# python train.py --name ttbar_particle_level --data-path /work/sesanche/ttH/UL/CMSSW_10_4_0/src/CMGTools/TTHAnalysis/macros/kk     --analysis ttbar_pl

from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import torch 
import uproot 

class dataset( IterableDataset ):
    def __init__( self, path, device):
        self.files=glob.glob(path)
        self.device=device
        self.length=0
        for fil in self.files:
            with uproot.open(fil) as f:
                self.length+=f['Friends'].num_entries

    def __len__(self):
        return self.length

    def __iter__(self):
        for fil in self.files:
            with uproot.open(fil) as f:
                tree = f['Friends']
                control_vars=torch.Tensor( tree.arrays(['parton_cnr_crn','parton_cnk_ckn','parton_crk_ckr'],library='pd').values ).to(self.device)
                weights  =torch.Tensor(tree.arrays(["weight_sm", "weight_ctgi_sm"], library='pd').values).to(self.device)
                variables=torch.Tensor(tree.arrays(['lp_px', 'lp_py', 'lp_pz',
                                                    'lm_px', 'lm_py', 'lm_pz',
                                                    'b1_px', 'b1_py', 'b1_pz',
                                                    'b2_px', 'b2_py', 'b2_pz',
                                                    'met_px', 'met_py'],library='pd').values).to(self.device)
            yield from zip(weights, control_vars, variables)



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

    def forward(self, x):
        
        cpx= torch.stack([-x[:,3], -x[:,4] , -x[:,5],    # -lep minus
                          -x[:,0], -x[:,1] , -x[:,2],    # -lep plus
                          -x[:,9], -x[:,10], -x[:,11],   # -b2 # this shouldnt add information but ok 
                          -x[:,6], -x[:,7] , -x[:,8],    # -b1 
                          -x[:,12],-x[:,13]          ],  # -met
                         dim=1)

        return self.main_module(x)-self.main_module(cpx)
