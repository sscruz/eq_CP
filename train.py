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
from data_networks import data_ttbar, ttbar_net
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
    parser.add_argument("--data-path"     , type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CP_equivariant/ttbar/ntuples", help="Path of the input dataset")
    args = parser.parse_args()

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

    training =data_ttbar( f'{data_path}/*.h5'     , device=args.device)
    test     =data_ttbar( f'{data_path}/test/*.h5', device='cpu')
    train_cpu=data_ttbar( f'{data_path}/*.h5'     , device='cpu')


    network=ttbar_net(args.device)
    dataloader = DataLoader( training, batch_size=args.batch_size) 

    test_loader  = DataLoader(test, batch_size=1000)
    train_loader = DataLoader(train_cpu, batch_size=1000)
    

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    
    def loss_func( weight, score, control):
        return torch.mean(weight[:,0]*(score[:,0]-(weight[:,1]/weight[:,0]))**2)



    train_loss_history=[]
    test_loss_history =[]
    for ep in range(args.epochs):

        loop=tqdm( dataloader)
        for weight, control, input_vars in loop:
            optimizer.zero_grad()

            # sergio score=network( input_vars )
            score=network( control[:,0].view(-1,1))
            loss=loss_func( weight, score, control)

            loss.backward()
            optimizer.step()

        with torch.no_grad():

            def do_end_of_era_processing( dataset, name ):
                loss=0; count=0
                regressed = defaultdict(list)
                truth     = defaultdict(list)
                sm        = defaultdict(list)
                binning = np.linspace(-1,1)

                for weight, control, input_vars in dataset:
                    #sergio score=network(input_vars)
                    score=network(control[:,0].view(-1,1))
                    loss +=loss_func( weight, score, control)*weight.shape[0] # multiply bc loss gives the average
                    count+=weight.shape[0]

                    if ep%5 == 0:

                        for var in range(control.shape[1]):
                            regressed[var].append( np.histogram( control[:,var], weights=(weight[:,0]*score[:,0]), bins=binning)[0])
                            truth    [var].append( np.histogram( control[:,var], weights=(weight[:,1])           , bins=binning)[0])
                            sm       [var].append( np.histogram( control[:,var], weights=(weight[:,0])           , bins=binning)[0])
                    if count >= 4e6:  # we only use 4M events for this
                        break

                if ep%5 == 0:
                    all_regressed = defaultdict(list)
                    all_truth     = defaultdict(list)
                    all_sm        = defaultdict(list)
                    for what in regressed:
                        
                        all_regressed[what] = sum(regressed[what])
                        all_truth    [what] = sum(truth[what])
                        all_sm       [what] = sum(sm[what])

                        all_regressed[what] = all_regressed[what] / all_sm[what]
                        all_truth    [what] = all_truth[what] / all_sm[what]
                        
                        plt.plot( (binning[1:]+binning[:-1])/2, all_regressed[what], label='Regressed')
                        plt.plot( (binning[1:]+binning[:-1])/2, all_truth[what]    , label='Truth')
                        plt.savefig( f'{args.name}/closure_{name}_var_{what}_epoch_{ep}.png')
                        plt.clf()


                return loss/count

            train_loss=do_end_of_era_processing(train_loader, 'train')
            test_loss =do_end_of_era_processing(test_loader , 'test')
            train_loss_history.append( train_loss )
            test_loss_history.append( test_loss )
            print(f"Epoch {ep:03d}: Loss (train) {train_loss:.5f}, Loss (test): {test_loss:.5f}")
            torch.save( network.state_dict(), f"{args.name}/state_{ep}.pt")
            torch.save( optimizer.state_dict(), f"{args.name}/optimizer_state_{ep}.pt")
            plt.plot( [x+1 for x in range(ep+1)], train_loss_history , label='Train')
            plt.plot( [x+1 for x in range(ep+1)], test_loss_history , label='Test')
            plt.savefig(f"{args.name}/training_history.png")
            plt.clf()
