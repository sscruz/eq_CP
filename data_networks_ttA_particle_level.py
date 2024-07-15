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
                control_vars=torch.Tensor( tree.arrays(["weight_ctzi_sm"],library='pd').values ).to(self.device)
                weights  =torch.Tensor(tree.arrays(["weight_sm", "weight_ctzi_sm"], library='pd').values).to(self.device)
                variables=torch.Tensor(tree.arrays(['lep_top_px', 'lep_top_py', 'lep_top_pz',
                                                    'random_b1_px', 'random_b1_py', 'random_b1_pz',
                                                    'random_b2_px', 'random_b2_py', 'random_b2_pz',
                                                    'random_j1_px', 'random_j1_py', 'random_j1_pz',
                                                    'random_j2_px', 'random_j2_py', 'random_j2_pz',
                                                    'lep_top_sign',
                                                    'met_px', 'met_py',
                                                    'photon_px', 'photon_py', 'photon_pz',

                                                ],library='pd').values).to(self.device)
            yield from zip(weights, control_vars, variables)



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
        
        cpx= torch.stack(
            [-x[:,0], -x[:,1] , -x[:,2],   # -lep 
             -x[:,6], -x[:,7],  -x[:,8],   # -b2 # this shouldnt add information but ok 
             -x[:,3], -x[:,4] , -x[:,5],   # -b1 
             -x[:,12], -x[:,13] , -x[:,14],   # -j2
             -x[:,9], -x[:,10] , -x[:,11],   # -j1
             -x[:,15], # -lep top sign
             -x[:,16],-x[:,17],  # -met
             -x[:,18],-x[:,19],-x[:,20], # -photon
             ],dim=1).to(self.device)

        return self.main_module(x)-self.main_module(cpx)
