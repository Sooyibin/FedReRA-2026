import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import cycle

'''import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os'''
import copy
from math import *
import random

import datetime

from model import *
from utils import *
# from vggmodel import *
# from resnetcifar import *
import arguments

args = arguments.get_args()
print(args)

def train_net_func(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu", global_dataloader=None, gb=None, **kwargs):
    # print(device)
    logger.info('Training network %s' % str(net_id))
    criterion = nn.CrossEntropyLoss().to(device)
    net.train()
    # net.register(gb)
    for epoch in range(epochs):
        epoch_loss_collector_shallow = []
        epoch_loss_collector_deep = []
        for data_batch in train_dataloader:

            images, labels = data_batch
            if(args.dataset == 'agnews'):
                images[0] = images[0].to(device)
            else:
                images = images.to(device)
            labels = labels.to(device)
            f = net.forward_f(images)
            pred = net.classify(f)
            if(args.shallow):
                loss = criterion(pred[0], labels) + criterion(pred[1], labels)
                epoch_loss_collector_shallow.append(criterion(pred[0], labels).item())
                epoch_loss_collector_deep.append(criterion(pred[1], labels).item())
            else:
                loss = criterion(pred[0], labels)
                epoch_loss_collector_shallow.append(0)
                epoch_loss_collector_deep.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_shallow = sum(epoch_loss_collector_shallow) / len(epoch_loss_collector_shallow)
        epoch_loss_deep = sum(epoch_loss_collector_deep) / len(epoch_loss_collector_deep)
        logger.info('Epoch: %d Loss: %f (shallow), %f (deep)' % (epoch, epoch_loss_shallow, epoch_loss_deep))
    
    logger.info(' ** Training complete **')
    # add your train_net_func code here
    n_dict = {"cifar10":10, "cifar100":100, "tinyimagenet":200, "fmnist":10, "agnews":4}
    nc = n_dict[args.dataset]
    sl = {}
    sf = {}
    if(args.dataset == "agnews"):
        gb[0] = gb[0].to(device)
    else:
        gb = gb.to(device)
    # bias = net.forward_f(gb)
    # print("gb shape: ", gb.shape)
    ct = np.zeros(nc)
    for c in range(nc):
        sf[c] = []
        
    for batch in train_dataloader:
        images, labels = batch
        if(args.dataset == 'agnews'):
            images[0] = images[0].to(device)
        else:
            images = images.to(device)
        labels = labels.numpy()
        ct[labels] += 1
        f = net.forward_f(images)
        for idx in range(labels.shape[0]):
            if(args.debias_shallow):
                sf[labels[idx]].append((f[args.layer][idx]).detach().cpu().numpy())
    if(args.debias_shallow):
        stt = {}
        for c in range(nc):
            if(ct[c] > 5):
                stt[c] = np.mean(sf[c], axis=0)  # np.mean(sf[c], axis=0)
                # stt[c] = net.classifier_1.state_dict()['weight'][c].detach().cpu().numpy()
                if(args.noise):
                    print("add noise")
                    s, pc = 5, 0.2
                    stt[c] = stt[c] * (1 - pc) + np.random.normal(loc=0.0, scale=s, size=stt[c].shape)
        sl[args.layer] = stt
    net.to('cpu')
    ei = sl, ct
    # return extra information if necessary; it's okay to return nothing
    return ei

def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        # for tmp in train_dataloader:
        for batch_idx, (x, target) in enumerate(train_dataloader):
            if(args.dataset == 'agnews'):
                x[0] = x[0].to(device)
                target = target.to(device,dtype=torch.int64)
            else:
                x, target = x.to(device), target.to(device)
            optimizer.zero_grad()
            # x[0].requires_grad = True
            # target.requires_grad = False
            target = target.long()
            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    logger.info(' ** Training complete **')

def train_net_fedprox(net_id, net,  train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu",mu=None, global_net = None, gb=None):

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            if(args.shallow):
                f = net.forward_f(x)
                pred = net.classify(f)
                loss = criterion(pred[0], target) + criterion(pred[1], target)

            #for fedprox, add one term
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
            loss += fed_prox_reg


            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    """
    num_dict = {"cifar10":10, "cifar100":100, "tinyimagenet":200}
    nc = num_dict[args.dataset]
    print(nc)
    sl = {}
    sf = {}
    deep_features = {}
    bias = net.forward_f(gb.to(device))
    ct = np.zeros(10)
    for c in range(nc):
        sf[c] = []
        deep_features[c] = []
    for batch in train_dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.numpy()
        f = net.forward_f(images)
        for idx in range(labels.shape[0]):
            ct[labels[idx]] += 1
            if(args.debias_shallow):
                sf[labels[idx]].append((f[1][idx] - torch.mean(bias[1], dim=0)).detach().cpu().numpy())
            if(args.debias_deep):
                deep_features[labels[idx]].append((f[2][idx] - torch.mean(bias[2], dim=0)).detach().cpu().numpy())
    if(args.debias_shallow):
        stt = {}
        for c in range(nc):
            if(len(sf[c]) > 30):
                stt[c] = np.mean(sf[c], axis=0)
        sl[1] = stt
    if(args.debias_deep):
        stt = {}
        for c in range(nc):
            if(len(deep_features[c]) > 30):
                stt[c] = np.mean(deep_features[c], axis=0)
        sl[2] = stt
    net.to('cpu')
    ei = sl
    return ei
    """

def train_net_scaffold(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu",gm=None, c_local=None, c_global=None, global_databatch=None):
    logger.info('Training network %s' % str(net_id))

    c_local = c_local[net_id]#to solve the problem moving this from local_train_net_scaffold

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    
    c_local.to(device)
    c_global.to(device)
    gm.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                if(args.shallow):
                    f = net.forward_f(x)
                    pred = net.classify(f)
                    loss = criterion(pred[0], target) + criterion(pred[1], target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    gm_para = gm.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (gm_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)
    sl = {}
    """
    num_dict = {"cifar10":10, "cifar100":100, "tinyimagenet":200, "fmnist":10}
    nc = num_dict[args.dataset]
    print(nc)
    sl = {}
    sf = {}
    deep_features = {}
    bias = net.forward_f(global_databatch.to(device))
    
    for c in range(nc):
        sf[c] = []
        deep_features[c] = []
    for batch in train_dataloader[0]:
        images, labels = batch
        images = images.to(device)
        labels = labels.numpy()
        f = net.forward_f(images)
        for idx in range(labels.shape[0]):
            if(args.debias_shallow):
                sf[labels[idx]].append((f[1][idx] - torch.mean(bias[1], dim=0)).detach().cpu().numpy())
            if(args.debias_deep):
                deep_features[labels[idx]].append((f[2][idx] - torch.mean(bias[2], dim=0)).detach().cpu().numpy())
    if(args.debias_shallow):
        stt = {}
        for c in range(nc):
            if(len(sf[c]) > 30):
                stt[c] = np.mean(sf[c], axis=0)
        sl[1] = stt
    if(args.debias_deep):
        stt = {}
        for c in range(nc):
            if(len(deep_features[c]) > 30):
                stt[c] = np.mean(deep_features[c], axis=0)
        sl[2] = stt
    """
    net.to('cpu')
    return c_delta_para, sl

def train_net_fednova(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu", arguments=None, gm=None,net_dataidx_map_in_train=None):

    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    gm.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    gm_para = gm.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(gm.state_dict())
    for key in norm_grad:
        #norm_grad[key] = (gm_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(gm_para[key]-net_para[key], a_i)
    net.to('cpu')
    logger.info(' ** Training complete **')

    #the part of getting len(train_dl_local.dataset()) is moved here. the "args" will be renamed arguments
    dataidxs = net_dataidx_map_in_train[net_id]
    if arguments.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(arguments.dataset, arguments.datadir, arguments.batch_size, 32, dataidxs, noise_level, net_id, arguments.n_parties-1)
    else:
        noise_level = arguments.noise / (arguments.n_parties - 1) * net_id
        train_dl_local, test_dl_local, _, _ = get_dataloader(arguments.dataset, arguments.datadir, arguments.batch_size, 32, dataidxs, noise_level)
    train_dataloader,test_dataloader, _, _ = get_dataloader(arguments.dataset, arguments.datadir, arguments.batch_size, 32)
    epochs = arguments.epochs
    return [a_i, norm_grad, len(train_dl_local.dataset)]

def train_net_moon(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer,  device="cpu",global_net=None, prev_model_pool = None,mu=None,temperature=None, arguments=None,gm=None,gb=None):

    previous_nets = [prev_model_pool[i][net_id] for i in range(len(prev_model_pool))]

    criterion = nn.CrossEntropyLoss().to(device)
    global_net.to(device)

    if arguments.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    
    cnt = 0
    cos=torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if arguments.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2-pro1, dim=1))

            elif arguments.loss == 'only_contrastive' or arguments.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)
                # print(loss2)

            if arguments.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                if(args.shallow):
                    pred = net.forward_method(x)
                    loss1 = criterion(pred[0], target) + criterion(pred[1], target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))


    if arguments.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    sl = {}
    """
    num_dict = {"cifar10":10, "cifar100":100, "tinyimagenet":200}
    nc = num_dict[args.dataset]
    print(nc)
    sl = {}
    sf = {}
    deep_features = {}
    bias = net.f.forward_f(gb.to(device))
    
    for c in range(nc):
        sf[c] = []
        deep_features[c] = []
    for batch in train_dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.numpy()
        f = net.f.forward_f(images)
        for idx in range(labels.shape[0]):
            if(args.debias_shallow):
                sf[labels[idx]].append((f[1][idx] - torch.mean(bias[1], dim=0)).detach().cpu().numpy())
            if(args.debias_deep):
                deep_features[labels[idx]].append((f[2][idx] - torch.mean(bias[2], dim=0)).detach().cpu().numpy())
    if(args.debias_shallow):
        stt = {}
        for c in range(nc):
            if(len(sf[c]) > 30):
                stt[c] = np.mean(sf[c], axis=0)
        sl[1] = stt
    if(args.debias_deep):
        stt = {}
        for c in range(nc):
            if(len(deep_features[c]) > 30):
                stt[c] = np.mean(deep_features[c], axis=0)
        sl[2] = stt
    """
    net.to('cpu')
    gm.to('cpu')
    logger.info(' ** Training complete **')
    ei = sl
    return ei

def train_net_fedlc(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu", class_cnt=[], tau=0.1):
    logger.info('Training network %s' % str(net_id))

    class_cnt = class_cnt[net_id]

    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x,target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                for i in range(len(class_cnt)):
                    if class_cnt[i] == 0:
                        out[:, i] -= 10000
                    else:
                        out[:, i] -= pow(class_cnt[i], -1/4) * tau

                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    logger.info(' ** Training complete **')

def train_net_fedrs(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer, device="cpu", class_cnt=0):
    logger.info('Training network %s' % str(net_id))

    class_cnt = class_cnt[net_id]

    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]
    
    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x,target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                for i in range(len(class_cnt)):
                    if class_cnt[i] == 0:
                        out[:, i] *= 0.5

                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    net.to('cpu')
    logger.info(' ** Training complete **')