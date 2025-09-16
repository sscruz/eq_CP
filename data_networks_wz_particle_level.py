# python train.py --name wz_recopz_feb27 --data-path minitrees/wz_training_withpz --analysis wz --data-format root --batch-size 1000 --lr 1e-4
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
                control_vars=torch.Tensor( tree.arrays(['parton_phi_z','parton_phi_w'],library='pd').values ).to(self.device)
                weights  =torch.Tensor(tree.arrays(["weight_sm", "weight_cwtil_sm"], library='pd').values).to(self.device)
                variables=torch.Tensor(tree.arrays(['lzp_px', 'lzp_py', 'lzp_pz',
                                                    'lzm_px', 'lzm_py', 'lzm_pz',
                                                    'lw_px', 'lw_py', 'lw_pz',
                                                    'met_px', 'met_py',
                                                    'lwsign'],library='pd').values).to(self.device)
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
        self.device=device

    def forward(self, x):
        cpx= torch.stack([-x[:,3], -x[:,4] , -x[:,5],           # -lep z minus 3-momenta
                          -x[:,0], -x[:,1] , -x[:,2],           # -lep z plus  3-momenta
                          -x[:,6], -x[:,7] , -x[:,8],           # -W 3-momenta
                          -x[:,9], -x[:,10],                    # -met
                          -x[:,11]],                            # -W sign
                         dim=1).to(self.device)
        return self.main_module(x)-self.main_module(cpx)
