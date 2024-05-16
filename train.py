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
    parser.add_argument("--num-threads"    ,  type=int  , default=16, help="Number of threads when running in cpu")
    parser.add_argument("--epochs"        ,  type=int  , default=200, help="Number of epochs")
    parser.add_argument("--prefetch"      ,  type=str  , default=None, help="Temporary directory to prefetch data")
    parser.add_argument("--data-format"   ,  type=str  , default='h5', help="Extension of input files")
    parser.add_argument("--data-path"     , type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/sesanche/CP_equivariant/ttbar/ntuples", help="Path of the input dataset")
    parser.add_argument("--analysis"     , type=str, default="ttbar", choices=['ttbar','ttbar_ideal','ttbar_withneutrinos', 'ttbb_godmode', 'ttZ_3l','ttZ_3l_v2','ttA_1l','ttW', 'ttbar_pl','ttA_pl', 'ww', 'wz','tzq_pl', 'ttz_pl', 'ttA_delphes', 'wz_delphes'], help="Analysis to run, defines dataset type and neural network")
    parser.add_argument("--load-model"     , type=str, default=None, help="Analysis to run, defines dataset type and neural network")
    parser.add_argument("--noequivariant"  , type=int, default=0, help="NN type")
    parser.add_argument("--no-plot"  ,  action='store_true',  help="Skip plotting" ) 

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
        if bool(args.noequivariant):
           from data_networks_ttbar_particle_level import network_noeq as network
           print("no equivariant")
        else:
           from data_networks_ttbar_particle_level import network
           print("equivariant")
    elif args.analysis == 'ttA_pl':
        from data_networks_ttA_particle_level import dataset, network
    elif args.analysis == 'ww':
        from data_networks_ww import dataset, network
    elif args.analysis == 'wz':
        from data_networks_wz_particle_level import dataset, network
    elif args.analysis == 'tzq_pl':
        from data_networks_tZq_particle_level import dataset, network
        max_value=0.4
    elif args.analysis == 'ttz_pl':
        from data_networks_ttZ_particle_level import dataset, network
    elif args.analysis == "ttA_delphes": 
        from data_networks_ttA_delphes import dataset, network
    elif args.analysis == "wz_delphes": 
        from data_networks_wz_delphes import dataset, network

    else:
        raise NotImplementedError(f"Option {args.analysis} not implemented")

    if args == 'cpu':
        torch.set_num_threads(args.num_threads)
        

    os.makedirs( args.name, exist_ok=True)
    if args.prefetch is not None:
        #dirpath = tempfile.mkdtemp(dir=args.prefetch)
        tempdir = tempfile.TemporaryDirectory(dir=args.prefetch)
        print(f"We are prefetching to {tempdir.name}")
        os.system(f'cp -r {args.data_path}/* {tempdir.name}/.')
        data_path=tempdir.name
    else:
        data_path=args.data_path

    print("Loading training ")
    training =dataset( f'{data_path}/*.{args.data_format}'     , device=args.device)
    print("Loading test ")
    test     =dataset( f'{data_path}/test/*.{args.data_format}', device='cpu')


    net=network(args.device)


    dataloader = DataLoader( training, batch_size=args.batch_size) 
    test_loader  = DataLoader(test, batch_size=1000)


    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.load_model is not None:
        model_state = torch.load(args.load_model, map_location=args.device)
        net.load_state_dict( model_state ) 
        model_dir, model_file=os.path.split( args.load_model) 
        opt_state   = torch.load( model_dir +  '/optimizer_' + model_file)
        optimizer.load_state_dict(opt_state)
    
    def loss_func( weight, score, control):
        return torch.mean(weight[:,0]*(score[:,0]-(weight[:,1]/weight[:,0]))**2)


    train_loss_history=[]
    test_loss_history =[]
    for ep in range(args.epochs):

        loop=tqdm( dataloader)
        loss_per_batch=[]
        for weight, control, input_vars in loop:
            optimizer.zero_grad()

            score=net( input_vars )
            loss=loss_func( weight, score, control)
            loss_per_batch.append( loss.item() )
            loss.backward()
            optimizer.step()
        #plt.hist( loss_per_batch, bins=100, alpha=0.5 ) 
        #print(loss_per_batch)
        #plt.show()
        #continue

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

                for_plot_true=torch.empty(0); for_plot_regress=torch.empty(0)
                for weight, control, input_vars in dataset:
                    score=net(input_vars)
                    loss +=loss_func( weight, score, control)*weight.shape[0] # multiply bc loss gives the average
                    count+=weight.shape[0]

                    if ep%5== 0 and not args.no_plot:
                        for_plot_true   =torch.cat( [for_plot_true   , weight[:,1]/weight[:,0]])
                        for_plot_regress=torch.cat( [for_plot_regress, score])
                        for var in range(control.shape[1]):
                            if hasattr(training, 'var_range'):
                                binning = np.linspace(training.var_range[var][0],training.var_range[var][1])
                            else:
                                binning = np.linspace(-1,1)
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
                       


                        
                        
                    

                if ep%5 == 0 and not args.no_plot:
                    all_regressed = defaultdict(list)
                    all_truth     = defaultdict(list)
                    all_sm        = defaultdict(list)
                    
                    plt.hist2d( for_plot_true.numpy(), for_plot_regress.flatten().numpy(), bins=40, range=[[-0.2,0.2],[-0.05,0.05]])
                    plt.savefig(f'{args.name}/{name}_2d_{ep}.png')
                    plt.clf()

                    plots_epoch={}
                    for what in regressed:
                        
                        all_regressed[what] = sum(regressed[what])
                        all_truth    [what] = sum(truth[what])
                        all_sm       [what] = sum(sm[what])

                        norm=200/np.sum(all_sm[what])
                        thebinning=(binnings[what][1:]+binnings[what][:-1])/2
                        plt.plot(thebinning, all_truth[what]*norm*10, label='Linear x 10')
                        plt.plot(thebinning, all_sm[what]   *norm, label='SM')
                        plt.legend()
                        plt.savefig(f'{args.name}/histogram_{name}_var_{what}_epoch_{ep}.png')
                        plt.clf()
                        
                        plots_epoch[f'histogram_{name}_{what}']={}
                        plots_epoch[f'histogram_{name}_{what}']['binning']=binnings[what].tolist()
                        plots_epoch[f'histogram_{name}_{what}']['SM']     =(all_sm[what]   *norm   ).tolist()
                        plots_epoch[f'histogram_{name}_{what}']['linear'] =(all_truth[what]*norm*10).tolist()
                        


                        all_regressed[what] = all_regressed[what] / all_sm[what]
                        all_truth    [what] = all_truth[what] / all_sm[what]
                        plt.plot( thebinning, all_regressed[what], label='Regressed')
                        plt.plot( thebinning, all_truth[what]    , label='Truth')
                        plt.savefig( f'{args.name}/closure_{name}_var_{what}_epoch_{ep}.png')
                        plt.clf()
                        plots_epoch[f'closure_{name}_{what}']={}
                        plots_epoch[f'closure_{name}_{what}']['binning']=binnings[what].tolist()
                        plots_epoch[f'closure_{name}_{what}']['regressed'] =all_regressed[what].tolist()
                        plots_epoch[f'closure_{name}_{what}']['truth']     =all_truth[what].tolist()
                    with open(f'{args.name}/plots_{name}_epoch_{ep}.txt', 'w') as f:
                        json.dump(plots_epoch, f)


                return loss/count

            train_loss=do_end_of_era_processing(dataloader, 'train')
            test_loss =do_end_of_era_processing(test_loader , 'test')
            train_loss_history.append( train_loss )
            test_loss_history.append( test_loss )
            print(f"Epoch {ep:03d}: Loss (train) {train_loss:.5e}, Loss (test): {test_loss:.5e}")
            torch.save( net.state_dict(), f"{args.name}/state_{ep}.pt")
            torch.save( optimizer.state_dict(), f"{args.name}/optimizer_state_{ep}.pt")
            torch.save( test_loss, f"{args.name}/testloss_{ep}.pt")
            torch.save( train_loss, f"{args.name}/trainloss_{ep}.pt")
            plt.plot( [x+1 for x in range(ep+1)], train_loss_history , label='Train')
            plt.plot( [x+1 for x in range(ep+1)], test_loss_history , label='Test')
            plt.legend()
            plt.yscale('log')
            plt.savefig(f"{args.name}/training_history.png")
            plt.clf()
