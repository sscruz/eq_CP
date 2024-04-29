import torch
import numpy as np 
from tqdm import tqdm
from torch import optim
import os 
import json
import matplotlib.pyplot as plt
from collections import defaultdict 
import os 
import tempfile 
import pdb 

from torch.utils.data import DataLoader

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--device"        ,  type=str  , default="cpu", help="Device (cpu, gpu number)")
    parser.add_argument("--name"          ,  type=str  , default="", help="Name of directory to store everything")
    parser.add_argument("--lr"            ,  type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch-size"    ,  type=int  , default=500, help="Batch size")
    parser.add_argument("--epochs"        ,  type=int  , default=1000, help="Number of epochs")
    parser.add_argument("--prefetch"      ,  type=str  , default=None, help="Temporary directory to prefetch data")
    parser.add_argument("--data-format"   ,  type=str  , default='h5', help="Extension of input files")
    parser.add_argument("--data-path"     , type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CP_equivariant/ttbar/ntuples", help="Path of the input dataset")
    parser.add_argument("--analysis"     , type=str, default="ttbar", choices=['ttbar','ttbar_ideal','ttbar_withneutrinos', 'ttbb_godmode', 'ttZ_3l','ttZ_3l_v2','ttA_1l','ttW', 'ttbar_pl','ttA_pl', 'ww'], help="Analysis to run, defines dataset type and neural network")
    parser.add_argument("--noequivariant"   ,  type=bool  , default=False, help="run on equivariant or non-equivariant ")

    args = parser.parse_args()
    max_value=0.1
    if args.analysis == 'ttbar':
        from data_networks_ttbar import dataset, network
    elif args.analysis == 'ttbar_ideal':
        from data_networks_ttbar_ideal import dataset, network
    elif args.analysis == 'ttbar_withneutrinos':
        from data_networks_ttbar_withneutrinos import dataset, network
    elif args.analysis == 'ttbb_godmode':
        from data_networks_ttbb_godmode import dataset, network
    elif args.analysis == 'ttZ_3l':
        from data_networks_ttZ import dataset, network
    elif args.analysis == 'ttZ_3l_v2':
        from data_networks_ttZ_v2 import dataset, network
    elif args.analysis == 'ttA_1l':
        from data_networks_ttA_1l import dataset, network
    elif args.analysis == 'ttW':
        from data_networks_ttW import dataset, network
        max_value=0.05
    elif args.analysis == 'ttbar_pl':
        from data_networks_ttbar_particle_level import dataset
        if args.noequivariant:
           from data_networks_ttbar_particle_level import network_noeq as network
           print("no equivariant")
        else:
           from data_networks_ttbar_particle_level import network
           print("equivariant")
    elif args.analysis == 'ttA_pl':
        from data_networks_ttA_particle_level import dataset, network
    elif args.analysis == 'ww':
        from data_networks_ww import dataset, network

    else:
        raise NotImplementedError(f"Option {args.analysis} not implemented")

    if args == 'cpu':
        torch.set_num_threads(16)
        

    os.makedirs( args.name, exist_ok=True)
    if args.prefetch is not None:
        #dirpath = tempfile.mkdtemp(dir=args.prefetch)
        tempdir = tempfile.TemporaryDirectory(dir=args.prefetch)
        print(f"We are prefetching to {tempdir.name}")
        os.system(f'cp -r {args.data_path}/* {tempdir.name}/.')
        data_path=tempdir.name
    else:
        data_path=args.data_path

    training =dataset( f'{data_path}/*.{args.data_format}'     , device=args.device)
    test     =dataset( f'{data_path}/test/*.{args.data_format}', device='cpu')
    train_cpu=dataset( f'{data_path}/*.{args.data_format}'     , device='cpu')


    net=network(args.device)

    dataloader = DataLoader( training, batch_size=args.batch_size) 

    test_loader  = DataLoader(test, batch_size=1000)
    train_loader = DataLoader(train_cpu, batch_size=1000)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    def loss_func( weight, score, control):
        return torch.mean(weight[:,0]*(score[:,0]-(weight[:,1]/weight[:,0]))**2)



    train_loss_history=[]
    test_loss_history =[]
    for ep in range(args.epochs):

        loop=tqdm( dataloader)
        for weight, control, input_vars in loop:
            optimizer.zero_grad()

            score=net( input_vars )
            loss=loss_func( weight, score, control)

            loss.backward()
            optimizer.step()

        with torch.no_grad():

            def do_end_of_era_processing( dataset, name ):
                loss=0; count=0
                regressed = defaultdict(list)
                truth     = defaultdict(list)
                sm        = defaultdict(list)
                
                symmetry_lin_plus  = []
                symmetry_sm_plus   = []
                symmetry_lin_minus = []
                symmetry_sm_minus  = []
                
                binnings  = {}
                binning = np.linspace(-1,1)

                for weight, control, input_vars in dataset:
                    score=net(input_vars)
                    loss +=loss_func( weight, score, control)*weight.shape[0] # multiply bc loss gives the average
                    count+=weight.shape[0]

                    if ep%1 == 0:

                        for var in range(control.shape[1]):
                            regressed[var].append( np.histogram( control[:,var], weights=(weight[:,0]*score[:,0]), bins=binning)[0])
                            truth    [var].append( np.histogram( control[:,var], weights=(weight[:,1])           , bins=binning)[0])
                            sm       [var].append( np.histogram( control[:,var], weights=(weight[:,0])           , bins=binning)[0])
                            binnings [var]=binning

                        bins=np.linspace(-max_value,max_value,26)
                        content,bins=np.histogram( score[:,0], weights=(weight[:,0]*score[:,0]), bins=bins)
                        regressed[control.shape[1]].append( content)
                        truth    [control.shape[1]].append( np.histogram( score[:,0], weights=(weight[:,1])           , bins=bins)[0])
                        sm       [control.shape[1]].append( np.histogram( score[:,0], weights=(weight[:,0])           , bins=bins)[0])
                        binnings [control.shape[1]]=bins
                       

                        
                        bins=np.linspace(0.,max_value,26)
                        symmetry_lin_plus  .append( np.histogram(score[:,0], weights=(weight[:,1]*np.where(score[:,0] > 0, 1,0 )), bins=bins)[0])
                        symmetry_sm_plus   .append( np.histogram(score[:,0], weights=(weight[:,0]*np.where(score[:,0] > 0, 1,0 )), bins=bins)[0])
                        symmetry_lin_minus .append( np.histogram(-score[:,0], weights=(weight[:,1]*np.where(score[:,0] < 0, -1,0)), bins=bins)[0])
                        symmetry_sm_minus  .append( np.histogram(-score[:,0], weights=(weight[:,0]*np.where(score[:,0] < 0, 1,0 )), bins=bins)[0])

                        
                        
                    

                if ep%1 == 0:
                    all_regressed = defaultdict(list)
                    all_truth     = defaultdict(list)
                    all_sm        = defaultdict(list)
                    
                    symmetry_lin_plus  = sum(symmetry_lin_plus)
                    symmetry_lin_minus = sum(symmetry_lin_minus)
                    symmetry_sm_plus   = sum(symmetry_sm_plus)
                    symmetry_sm_minus  = sum(symmetry_sm_minus)

                    bins=np.linspace(0.,0.1,26)
                    plt.plot((bins[1:]+bins[:-1])/2, symmetry_lin_plus*10, label='Linear (positive) x 10')
                    plt.plot((bins[1:]+bins[:-1])/2, symmetry_lin_minus*10, label='Linear (negative) x 10')
                    plt.plot((bins[1:]+bins[:-1])/2, symmetry_sm_plus, label='SM (positive) ')
                    plt.plot((bins[1:]+bins[:-1])/2, symmetry_sm_minus, label='SM (negative) ')
                    plt.legend()
                    plt.savefig(f'{args.name}/symmetry_{name}_var_epoch_{ep}.png')
                    plt.clf()


                    for what in regressed:
                        
                        all_regressed[what] = sum(regressed[what])
                        all_truth    [what] = sum(truth[what])
                        all_sm       [what] = sum(sm[what])

                        norm=200/np.sum(all_sm[what])
                        plt.plot((binnings[what][1:]+binnings[what][:-1])/2, all_truth[what]*norm*10, label='Linear x 10')
                        plt.plot((binnings[what][1:]+binnings[what][:-1])/2, all_sm[what]   *norm, label='SM')
                        plt.legend()
                        plt.savefig(f'{args.name}/histogram_{name}_var_{what}_epoch_{ep}.png')
                        plt.clf()

                        all_regressed[what] = all_regressed[what] / all_sm[what]
                        all_truth    [what] = all_truth[what] / all_sm[what]
                        plt.plot( (binnings[what][1:]+binnings[what][:-1])/2, all_regressed[what], label='Regressed')
                        plt.plot( (binnings[what][1:]+binnings[what][:-1])/2, all_truth[what]    , label='Truth')
                        plt.ylabel("Linear/SM")
                        plt.xlabel(training.name_cvaris(what))
                        plt.savefig( f'{args.name}/closure_{name}_var_{what}_epoch_{ep}.png')
                        plt.clf()


                return loss/count

            train_loss=do_end_of_era_processing(train_loader, 'train')
            test_loss =do_end_of_era_processing(test_loader , 'test')
            train_loss_history.append( train_loss )
            test_loss_history.append( test_loss )
            print(f"Epoch {ep:03d}: Loss (train) {train_loss:.5e}, Loss (test): {test_loss:.5e}")
            torch.save( net.state_dict(), f"{args.name}/state_{ep}.pt")
            torch.save( optimizer.state_dict(), f"{args.name}/optimizer_state_{ep}.pt")
            print(type(test_loss),type(optimizer))
            torch.save( test_loss, f"{args.name}/testloss_{ep}.pt")
            torch.save( train_loss, f"{args.name}/trainloss_{ep}.pt")
            plt.plot( [x+1 for x in range(ep+1)], train_loss_history , label='Train')
            plt.plot( [x+1 for x in range(ep+1)], test_loss_history , label='Test')
            plt.legend()
            plt.yscale('log')
            plt.savefig(f"{args.name}/training_history.png")
            plt.clf()
