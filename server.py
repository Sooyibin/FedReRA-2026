import copy
from math import *
import numpy as np
import random

from model import *
from utils import *
# from vggmodel import *
# from resnetcifar import *
import arguments
import experiments

args = arguments.get_args()
# you should upload this file and make sure that it fits in our framework

# the first part is to initialize your server-side variables before local-train. the variables therefore are kept within the function to avoid memory overuse
# to use the variables in experiments.py, write experiments.xxx

def init_server_func(args,nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):# you can change this name
    print('use this function to initialize the server')
    ei = []
    if args.aux_dataset == "tinyimagenet":
        maxnum = 100000
    elif args.aux_dataset == "people":
        maxnum = 25260
    elif args.aux_dataset == "face":
        maxnum = 450
    elif args.aux_dataset == "agnews":
        maxnum = 1500

    else:
        maxnum = 50000
    if(args.aux_dataset == "random"):
        g = torch.rand((args.num_auxdata,3,32,32), dtype=torch.float32).to(args.dvc)
        tdl = DataLoader(g, args.num_auxdata, True)
        gd = next(iter(tdl)).to(args.dvc)
    else:
        print(args.aux_dataset)
        grey = 1 if args.dataset == 'fmnist' else 0
        did = np.random.choice(np.arange(maxnum), args.num_auxdata)
        tdl, _, _, _ = get_dataloader(args.aux_dataset, args.datadir, 384, 32, did, aux=grey)
        tdl_full, _, _, _ = get_dataloader(args.aux_dataset, args.datadir, args.num_auxdata, 32, did, aux=grey)
        gd = next(iter(tdl_full))[0]

    

    labels = args.aux_label.split(" ")
    if(labels[0].isdigit()):
        print("args.aux_label", args.aux_label)
        _, _, trds, _ = get_dataloader('tinyimagenet', args.datadir, args.num_auxdata, 32)
        gd = torch.empty((args.num_auxdata, 3, 32, 32), dtype=torch.float32)
        labels = [int(label) for label in labels]
        labels, counts = np.unique(np.random.choice(labels, args.num_auxdata, replace=True, p=None), return_counts=True)
        cd = dict(zip(labels, counts))
        idx = 0
        for i in range(len(trds)):
            if(idx < args.num_auxdata and trds[i][1] in cd.keys() and cd[trds[i][1]] > 0):
                gd[idx] = trds[i][0]
                cd[trds[i][1]] -= 1
                idx += 1
    
    gm = gm.to(args.dvc)
    af = gm.forward_f(gd.to(args.dvc))[args.layer] - torch.mean(gm.forward_f(gd.to(args.dvc))[args.layer], dim=0)
    normalized = F.normalize(af, p=2, dim=1)
    sm = torch.mm(normalized, normalized.t())
    # print(sm)

    return [tdl, gd, sm]

def init_server_scaffold(args,nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):
    c_nets,_,_ = experiments.init_nets(args.net_config,args.dropout_p,args.n_parties,args)
    c_globals, _, _ = experiments.init_nets(args.net_config,0,1,args)
    c_global = c_globals[0]
    c_global_para = c_global.state_dict()
    for net_id,net in c_nets.items():
         net.load_state_dict(c_global_para)
    

    did = np.random.choice(np.arange(100000), args.num_auxdata)
    grey = 1 if args.dataset == 'fmnist' else 0
    train_dl_local_full, _, _, _ = get_dataloader(args.aux_dataset, args.datadir, args.num_auxdata, 32, did, aux=grey)
    gd = next(iter(train_dl_local_full))[0]

    return [c_nets,c_global, gd]

def init_server_fednova(args,nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):
    d_list = [copy.deepcopy(gm.state_dict()) for i in range(args.n_parties)]
    d_total_round = copy.deepcopy(gm.state_dict())
    for i in range(args.n_parties):
        for kk in d_list[i]:
            d_list[i][kk] = 0
    for kk in d_total_round:
        d_total_round[kk] = 0
    return [d_list,d_total_round]

def init_server_moon(args,nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):
    if args.is_same_initial:
        for net_id, net in nts.items():
            net.load_state_dict(gp)

    old_nets_pool = []
    old_nets = copy.deepcopy(nts)
    for _, net in old_nets.items():
        net.eval()
        for param in net.parameters():
            param.requires_grad = False
    
    did = np.random.choice(np.arange(100000), args.num_auxdata)
    train_dl_local_full, _, _, _ = get_dataloader('tinyimagenet', args.datadir, args.num_auxdata, 32, did)
    gd = next(iter(train_dl_local_full))[0]
    return [old_nets_pool,old_nets,gd]

def init_server_fedlc(args, nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):
    # X_train, y_train, X_test, y_test, ndm, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    # arr = np.arange(args.n_parties)
    # slct = arr[:int(args.n_parties * args.sample)]
    if args.dataset == "cifar10":
        K=10
    elif args.dataset == "cifar100":
        K=100

    client_cnt_all = []
    for net_id, net in nts.items():
        class_cnt = [0 for i in range(K)]
        for kk in traindata_cls_counts[net_id]:
            class_cnt[kk] = traindata_cls_counts[net_id][kk]
        client_cnt_all.append(class_cnt)
    return client_cnt_all

# fedrs requires the same class_cnt as fedlc
def init_server_fedrs(args, nts, local_model_meta_data, layer_type,gms, gm_meta_data, global_layer_type,gm,gp, traindata_cls_counts):
    # X_train, y_train, X_test, y_test, ndm, traindata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    if args.dataset == "cifar10":
        K=10
    elif args.dataset == "cifar100":
        K=100

    client_cnt_all = []
    for net_id, net in nts.items():
        class_cnt = [0 for i in range(K)]
        for kk in traindata_cls_counts[net_id]:
            class_cnt[kk] = traindata_cls_counts[net_id][kk]
        client_cnt_all.append(class_cnt)
    return client_cnt_all

# the second part is for server aggregation

def server_aggregate_func(nts,slct,args,dl_dict,ndm,tdg,dvc,gm,gp, r, ei=None):
    print('use this function to perform server aggregation in every communication round')
    global_dl = ei[0]
    gb = ei[1]

    if(args.dataset == "agnews"):
        gb[0] = gb[0].to(dvc)
    else:
        gb = gb.to(dvc)
    sm_old = ei[2]

    gi = experiments.local_train_net('method',nts,slct,args,dl_dict,ndm,tdg,dvc, global_dataloader=global_dl, gb=gb)
    
    af = gm.forward_f(gb)[args.layer] - torch.mean(gm.forward_f(gb)[args.layer], dim=0)
    normalized = F.normalize(af, p=2, dim=1)
    sm = torch.mm(normalized, normalized.t())
    sd = F.mse_loss(sm_old, sm)
    print(sd)
    if(args.dataset == "agnews"):
        gb[0] = gb[0].to(dvc)
    else:
        gb = gb.to(dvc)
        
    op = copy.deepcopy(gm.classifier_1.state_dict())['weight']
    ob = gm.forward_f(gb)
    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]
    # print(faf)
    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)
    # gm.classifier_1.load_state_dict(op)
    gm.to(dvc)
    nb = gm.forward_f(gb)
    gm.register(gb)
    nd = {"cifar10":10, "cifar100":100, "tinyimagenet":200, "fmnist":10, "agnews":4}
    if args.model == "avgnet":
        dd = {0:6272, 1:1600, 2:512}
        dim = dd[args.layer]
    elif args.model == "resnet":
        if args.dataset in ("cifar10", "cifar100"):
            dd = {0:16384, 1:16384, 2:8192, 3:4096, 4:64}
            dim = dd[args.layer]
        elif args.dataset == "tinyimagenet":
            dim = 4096
    elif args.model == "fedfanet":
        if args.dataset in ("cifar10", "cifar100"):
            dim = 64*5*5
        elif args.dataset == "tinyimagenet":
            dim = 4096
    elif args.model == "fedetfnet":
        if args.dataset in ("cifar10", "cifar100"):
            dim = 16384
        elif args.dataset == "tinyimagenet":
            dim = 16384
    elif args.model == "fasttext":
        dim = 32
    elif args.model == 'vgg16':
        dim = 6144
    if (sd > args.comm_thres or sd < 1e-10):
        info_3 = sm
        nc = nd[args.dataset]
        if(args.debias_shallow):
            bses = {}
            for i in slct:
                bses[i] = torch.mean(nts[i].to(dvc).forward_f(gb)[args.layer], dim=0).detach().cpu().numpy()
                # nts[i].register(gb)
            cl1 = [[] for c in range(nc)]
            # freq = [[] for c in range(nc)]
            print(slct, gi.keys())
            for c in range(nc):
                clct = False
                for i in slct:
                    if(c in gi[i][0][args.layer].keys()):
                        cl1[c].append(gi[i][0][args.layer][c] - bses[i])
                        # freq[c].append(gi[i][1][c])
                        clct = True
                if not clct:
                    print(c)
                    new = op[c] - torch.mean(ob[args.layer], dim=0) + torch.mean(nb[args.layer], dim=0)
                    cl1[c].append(new.detach().cpu().numpy())
            ctrs = torch.tensor(np.array([np.average(np.array(cl1[i]), axis=0) for i in range(nc)])).to(dvc)
            norm = torch.norm(ctrs, p=2, dim=1).expand(dim, ctrs.shape[0]).T
            ctrs = ctrs / norm
            stdct = gm.classifier_1.state_dict()
            stdct['weight'] = ctrs.to(dvc) 
            gm.classifier_1.load_state_dict(stdct)
    else:
        info_3 = sm_old
    
    return ei[0], ei[1], info_3

def server_aggregate_fedavg(nts,slct,args,dl_dict,ndm,tdg,dvc,gm,gp,ei=None):
    experiments.local_train_net('fedavg',nts,slct,args,dl_dict,ndm,tdg,dvc)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)

def server_aggregate_fedprox(nts,slct,args,dl_dict,ndm,tdg,dvc,gm,gp,ei=None):
    gb = ei[1]
    gi = experiments.local_train_net('fedprox',nts,slct,args,dl_dict,ndm,tdg,dvc,global_net = gm,mu=args.mu, gb=gb)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)

    gm.to(dvc)
    # gm.register(gb.to(dvc))

    nd = {"cifar10":10, "cifar100":100, "tinyimagenet":200}
    nc = nd[args.dataset]
    if(args.debias_shallow):
        cl1 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in gi[i][1].keys()):
                    print(i, c)
                    cl1[c].append(gi[i][1][c])
        ctrs = torch.tensor(np.array([np.average(np.array(cl1[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(1600, ctrs.shape[0]).T
        ctrs = ctrs / norm
        print(ctrs.shape, torch.norm(ctrs[0], p=2))
        stdct = gm.classifier_1.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.classifier_1.load_state_dict(stdct)
    if(args.debias_deep):
        center_list_2 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in gi[i][2].keys()):
                    center_list_2[c].append(gi[i][2][c])
        ctrs = torch.tensor(np.array([np.average(np.array(center_list_2[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(512, ctrs.shape[0]).T
        ctrs = ctrs / norm
        stdct = gm.classifier_2.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.classifier_2.load_state_dict(stdct)

def server_aggregate_scaffold(nts,slct,args,dl_dict,ndm,tdg,dvc,gm,gp, ei=None):
    gb = ei[2]
    c_nets, c_global = ei[0], ei[1]

    c_delta_para_all=experiments.local_train_net('scaffold',nts,slct,args,dl_dict,ndm,tdg,dvc,gm=gm,c_local=c_nets,c_global=c_global,gd=gb)

    total_delta = copy.deepcopy(gm.state_dict())
    for kk in total_delta:
        total_delta[kk] = 0.0
    c_global.to(dvc)
    gm.to(dvc)

    for kk in c_delta_para_all:
        c_delta_para = c_delta_para_all[kk][0]
        for local_key in total_delta:
            total_delta[local_key] += c_delta_para[local_key]
    for kk in total_delta:
        total_delta[kk] /= args.n_parties
    c_global_para = c_global.state_dict()
    for kk in c_global_para:
        if c_global_para[kk].type() == 'torch.LongTensor':
            c_global_para[kk] += total_delta[kk].type(torch.LongTensor)
        elif c_global_para[kk].type() == 'torch.cuda.LongTensor':
            c_global_para[kk] += total_delta[kk].type(torch.cuda.LongTensor)
        else:
            c_global_para[kk] += total_delta[kk]
    c_global.load_state_dict(c_global_para)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)
    gm.to(dvc)
    # gm.register(gb.to(dvc))

    nd = {"cifar10":10, "cifar100":100, "tinyimagenet":200, "fmnist":10}
    nc = nd[args.dataset]
    if(args.debias_shallow):
        cl1 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in c_delta_para_all[i][1][1].keys()):
                    print(i, c)
                    cl1[c].append(c_delta_para_all[i][1][1][c])
        ctrs = torch.tensor(np.array([np.average(np.array(cl1[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(1600, ctrs.shape[0]).T
        ctrs = ctrs / norm
        print(ctrs.shape, torch.norm(ctrs[0], p=2))
        stdct = gm.classifier_1.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.classifier_1.load_state_dict(stdct)
    if(args.debias_deep):
        center_list_2 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in c_delta_para_all[i][1][2].keys()):
                    center_list_2[c].append(c_delta_para_all[i][1][2][c])
        ctrs = torch.tensor(np.array([np.average(np.array(center_list_2[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(512, ctrs.shape[0]).T
        ctrs = ctrs / norm
        stdct = gm.classifier_2.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.classifier_2.load_state_dict(stdct)

def server_aggregate_fednova(nts,slct,args,dl_dict,ndm,tdg,dvc,gm, gp, ei=None):
    gi = experiments.local_train_net('fednova', nts, slct, args, dl_dict,
                                              ndm, tdg, dvc,
                                              arguments=args, gm=gm,
                                              net_dataidx_map_in_train=ndm)
    list_gi = list(gi.values()) # transform gi (dict) to list
    all_lists = [item for sublist in list_gi for item in sublist] # flatten list_gi (list embedded by list, please don't blame me, please blame someone asking me to modify again and again)
    a_list = [all_lists[3*i] for i in range(len(all_lists)//3)]
    d_list = [all_lists[3*i+1] for i in range(len(all_lists)//3)]
    n_list = [all_lists[3*i+2] for i in range(len(all_lists)//3)]
    total_n = sum(n_list)
    d_total_round = copy.deepcopy(gm.state_dict())
    for kk in d_total_round:
        d_total_round[kk] = 0.0

    for i in range(len(slct)):
        d_para = d_list[i]
        for kk in d_para:
            d_total_round[kk] += d_para[kk] * n_list[i] / total_n

    # update global model
    coeff = 0.0
    for i in range(len(slct)):
        coeff = coeff + a_list[i] * n_list[i]/total_n

    updated_model = gm.state_dict()
    for kk in updated_model:
        if updated_model[kk].type() == 'torch.LongTensor':
            updated_model[kk] -= (coeff * d_total_round[kk]).type(torch.LongTensor)
        elif updated_model[kk].type() == 'torch.cuda.LongTensor':
            updated_model[kk] -= (coeff * d_total_round[kk]).type(torch.cuda.LongTensor)
        else:
            updated_model[kk] -= coeff * d_total_round[kk]
    gm.load_state_dict(updated_model)

def server_aggregate_moon(nts,slct,args,dl_dict,ndm,tdg,dvc,gm,gp, ei=None):
    old_nets_pool = ei[0]
    old_nets = ei[1]
    gb = ei[2]


    gi = experiments.local_train_net('moon',nts,slct,args,dl_dict,ndm,tdg,dvc=dvc,global_net=gm,prev_model_pool=old_nets_pool,mu=args.mu,temperature=args.temperature,arguments=args,gm=gm,gb=gb)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)

    old_nets = copy.deepcopy(nts)
    for _, net in old_nets.items():
        net.eval()
        for param in net.parameters():
            param.requires_grad = False
    if len(old_nets_pool) < 1:
        old_nets_pool.append(old_nets)
    else:
        old_nets_pool[0] = old_nets

    gm.to(dvc)
    # gm.register(gb.to(dvc))
    nd = {"cifar10":10, "cifar100":100, "tinyimagenet":200}
    nc = nd[args.dataset]
    if(args.debias_shallow):
        cl1 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in gi[i][1].keys()):
                    print(i, c)
                    cl1[c].append(gi[i][1][c])
        ctrs = torch.tensor(np.array([np.average(np.array(cl1[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(1600, ctrs.shape[0]).T
        ctrs = ctrs / norm
        print(ctrs.shape, torch.norm(ctrs[0], p=2))
        stdct = gm.l2.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.l2.load_state_dict(stdct)
    if(args.debias_deep):
        center_list_2 = [[] for c in range(nc)]
        for c in range(nc):
            for i in range(args.n_parties):
                if(c in gi[i][2].keys()):
                    center_list_2[c].append(gi[i][2][c])
        ctrs = torch.tensor(np.array([np.average(np.array(center_list_2[i]), axis=0) for i in range(nc)])).to(dvc)
        norm = torch.norm(ctrs, p=2, dim=1).expand(512, ctrs.shape[0]).T
        ctrs = ctrs / norm
        stdct = gm.l3.state_dict()
        stdct['weight'] = ctrs.to(dvc)
        gm.l3.load_state_dict(stdct)


def server_aggregate_fedlc(nts,slct,args,ndm,tdg,dvc,gm,gp, ei=None):
    class_cnt = ei
    experiments.local_train_net('fedlc',nts,slct,args,ndm,tdg,dvc, class_cnt=class_cnt)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)

def server_aggregate_fedrs(nts,slct,args,ndm,tdg,dvc,gm,gp, ei=None):
    class_cnt = ei
    experiments.local_train_net('fedrs',nts,slct,args,ndm,tdg,dvc, class_cnt=class_cnt)

    # update global model
    tdp = sum([len(ndm[r]) for r in slct])
    faf = [len(ndm[r]) / tdp for r in slct]

    for idx in range(len(slct)):
        npara = nts[slct[idx]].cpu().state_dict()
        if idx == 0:
            for kk in npara:
                gp[kk] = npara[kk] * faf[idx]
        else:
            for kk in npara:
                gp[kk] += npara[kk] * faf[idx]
    gm.load_state_dict(gp)