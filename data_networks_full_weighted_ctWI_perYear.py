# python train.py --name ttbar_particle_level --data-path /work/sesanche/ttH/UL/CMSSW_10_4_0/src/CMGTools/TTHAnalysis/macros/kk     --analysis ttbar_pl

from torch.utils.data import Dataset, IterableDataset
import torch.nn as nn
import glob 
import torch 
import uproot 

class dataset( Dataset ):
    def __init__( self, path, device):
        self.files=glob.glob(path)
        print("Using files ", self.files)
        self.device=device
        self.control_vars=torch.Tensor( uproot.concatenate( [f"{path}:Events"], ["_input_weight_lin_ctWI/_input_weight_sm"],library='pd').values).to(self.device)
        self.weights     =torch.Tensor( uproot.concatenate( [f"{path}:Events"], ["mcSampleWeight*mcWeight*sm_weight*resampling_weight", "mcSampleWeight*mcWeight*_input_weight_lin_ctWI*resampling_weight"],library='pd').values).to(self.device)
        
        self.variables=torch.Tensor( uproot.concatenate( [f"{path}:Events"], ['_input_lw_px', '_input_lw_py', '_input_lw_pz',
                                                                              '_input_lw_q', 
                                                                              '_input_met_px', '_input_met_py',
                                                                              '_input_zplus_px', '_input_zplus_py', '_input_zplus_pz',
                                                                              '_input_zminus_px', '_input_zminus_py', '_input_zminus_pz',
                                                                              '_input_j1_px', '_input_j1_py', '_input_j1_pz', '_input_j1_btag',
                                                                              '_input_j2_px', '_input_j2_py', '_input_j2_pz', '_input_j2_btag',
                                                                              '_input_j3_px', '_input_j3_py', '_input_j3_pz', '_input_j3_btag',
                                                                              '_input_j4_px', '_input_j4_py', '_input_j4_pz', '_input_j4_btag',
                                                                              '_input_j5_px', '_input_j5_py', '_input_j5_pz', '_input_j5_btag',
                                                                              'eraFlag', 

                                                                              ],library='pd').values).to(self.device)

        

        self.length=self.control_vars.shape[0]


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.weights[i,:], self.control_vars[i,:], self.variables[i,:]




class network(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(33,160),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(160, 160),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(160, 80),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(80, 20),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(20, 1, bias=False),
        )
        self.main_module.to(device)

    def forward(self, x):
        
        cpx= torch.stack(
            [
                -x[:,0], -x[:,1] , -x[:,2],   # -lep
                -x[:,3],                      # -qlep
                -x[:,4], -x[:,5],             # -met
                -x[:,9], -x[:,10] , -x[:,11],   # -zminus
                -x[:,6], -x[:,7],  -x[:,8],     # -zplus
                -x[:,12], -x[:,13],  -x[:,14], x[:,15],     # -j1, btag is not opposed
                -x[:,16], -x[:,17],  -x[:,18], x[:,19],     # -j2, btag is not opposed
                -x[:,20], -x[:,21],  -x[:,22], x[:,23],     # -j3, btag is not opposed
                -x[:,24], -x[:,25],  -x[:,26], x[:,27],     # -j4, btag is not opposed
                -x[:,28], -x[:,29],  -x[:,30], x[:,31],     # -j5, btag is not opposed
                x[:,32],
            ],dim=1)

        return self.main_module(x)-self.main_module(cpx)
