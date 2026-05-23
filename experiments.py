import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
import time  
import datetime
# from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *

import trainnet
import arguments
import server


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2
    elif args.dataset == "agnews":
        n_classes = 4

    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            if args.dataset in ("cifar10", "cifar100") and args.model == 'resnet':
                add = "-cifar"
            if args.dataset == "tinyimagenet" and args.model == 'resnet':
                add = "-tiny"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model + add, args.out_dim, n_classes, args.debiased_inf, net_configs, args=args)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    if args.dataset in ("cifar10", "cifar100"):
                        net = ResNet_cifar(
                        resnet_size=20,
                        group_norm_num_groups=2,
                        freeze_bn=True,
                        freeze_bn_affine=True,
                        num_classes=n_classes,
                        args=args
                    )
                    elif args.dataset == 'tinyimagenet':
                        net = ResNet_imagenet(
                        resnet_size=18,
                        group_norm_num_groups=2,
                        freeze_bn=True,
                        freeze_bn_affine=True,
                        args=args
                    )
                    # for name, parameter in net.named_parameters():
                    #     print(f"{name} has {parameter.numel()} parameters")
                elif args.model == "vgg16":
                    net = VGG16(args, num_classes=100)
                elif args.model == "avgnet":
                    device = torch.device(args.device)
                    if args.dataset == 'cifar100':
                        net = FedAvgNet(args, num_classes=100, dim=1600)
                    elif args.dataset == 'tinyimagenet':
                        net = FedAvgNet(args, num_classes=200, dim=1600)
                    elif args.dataset == 'fmnist':
                        net = FedAvgNet(args, in_features=1, num_classes=10, dim=1600)
                    else: 
                        net = FedAvgNet(args, dim=1600)
                elif args.model == "convnet":
                    device = torch.device(args.device)
                    if args.dataset == 'cifar100':
                        net = ConvNet()
                    elif args.dataset == 'tinyimagenet':
                        net = ConvNet()
                    else: 
                        net = ConvNet()
                elif args.model == "fedfanet":
                    net = FedFANet(args, args.dataset)
                elif args.model == "fedetfnet":
                    if(args.dataset == 'cifar10'):
                        net = ResNet20(num_classes=10, args=args)
                    elif(args.dataset == 'cifar100'):
                        net = ResNet20(num_classes=100, args=args)
                    elif(args.dataset == 'tinyimagenet'):
                        net = ResNet18(args=args)
                elif args.model == "fasttext":
                    emb_dim = 32
                    vocab_size = 98635
                    net = fastText(args, hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=4).to(args.device)
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net
            print(net)
            for name, parameters in net.named_parameters():
                print(name, ':', parameters.size())

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def local_train_net(alg, nets, selected, args, dl_dict, net_dataidx_map, test_dl=None, device="cpu", **kwargs):
    global_info = {}

    if alg == 'fedavg':
        train_net_func = trainnet.train_net_fedavg
    elif alg == 'fedprox':
        train_net_func = trainnet.train_net_fedprox
    elif alg == 'scaffold':
        train_net_func = trainnet.train_net_scaffold
    elif alg == 'moon':
        train_net_func = trainnet.train_net_moon
    elif alg == 'fednova':
        train_net_func = trainnet.train_net_fednova
    elif alg == 'fedlc':
        train_net_func = trainnet.train_net_fedlc
    elif alg == 'fedrs':
        train_net_func = trainnet.train_net_fedrs
    else:
        train_net_func = trainnet.train_net_func

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.rho,
                                  weight_decay=args.reg)

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        train_dl_local, test_dl_local = dl_dict[net_id]

        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        # Here we invoke train_net_func and train the models, and gather whatever information you need
        # train_net_func is allowed to return an info (pack all the returns in the info) and they will be gathered in Info
        local_info = train_net_func(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, optimizer, device=device,
                                    **kwargs)

        try:  # In case train_net_xxx returns None
            global_info[net_id] = local_info
        except:
            pass
    return global_info


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(2024)
    random.seed(2024)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset, datadir, logdir,
                                                                                             partition, n_parties,
                                                                                             beta=beta)

    return net_dataidx_map

def client_dataloader(X_train, y_train, X_test, y_test, batch_size, dataidxs):
    print(zip(*X_train[dataidxs]))
    X_train, X_train_lens = list(zip(*X_train[dataidxs]))
    X_train = torch.Tensor(X_train).type(torch.int64)
    X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
    y_train = torch.Tensor(y_train[dataidxs]).type(torch.int64)
    train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]

    X_test, X_test_lens = list(zip(*X_test))
    X_test = torch.Tensor(X_test).type(torch.int64)
    X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
    y_test = torch.Tensor(y_test).type(torch.int64)
    test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]

    train_dl = DataLoader(train_data, batch_size, drop_last=False, shuffle=True)
    test_dl = DataLoader(test_data, batch_size, drop_last=True, shuffle=True)
    return train_dl, test_dl

if __name__ == '__main__':
    
    args = arguments.get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    # print(list(zip(X_train[net_dataidx_map[0]])))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,
                                                                                      args.batch_size, 32)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    train_all_in_list = []
    test_all_in_list = []
    print('alg is', args.alg)

    start_time = time.time()
    inference_time = datetime.timedelta()

    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        
        dl_dict = {}
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            noise_level = args.noise
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
            # train_dl_local, test_dl_local = client_dataloader(X_train, y_train, X_test, y_test, args.batch_size, dataidxs)
            dl_dict[net_id] = (train_dl_local, test_dl_local)
        
        
        accs = np.empty(args.comm_round, dtype=np.float32)
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            # lr_scheduler(round, args)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            server.server_aggregate_fedavg(nets, selected, args, dl_dict, net_dataidx_map, test_dl_global, device, global_model,
                                           global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc, avg_time = compute_accuracy(global_model, train_dl_global, device=device, dataset='cifar10')
            test_acc, conf_matrix, avg_time = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device, dataset='cifar10')
            inference_time += avg_time
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            accs[round] = test_acc
            print(round, test_acc)
        print(accs)
        np.save(args.accdir, accs)

    else:
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        # define extra information by calling init_server_func()
        extra_info = server.init_server_func(args, nets, local_model_meta_data, layer_type, global_models,
                                             global_model_meta_data, global_layer_type, global_model, global_para,
                                             traindata_cls_counts)
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)
        dl_dict = {}
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            noise_level = args.noise

            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                dataidxs, noise_level)
            # train_dl_local, test_dl_local = client_dataloader(X_train, y_train, X_test, y_test, args.batch_size, dataidxs)
            dl_dict[net_id] = (train_dl_local, test_dl_local)
        accs = np.empty(args.comm_round, dtype=np.float32)
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            # lr_scheduler(round, args)

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # call server_aggregate_func
            extra_info = server.server_aggregate_func(nets, selected, args, dl_dict, net_dataidx_map, test_dl_global, device, global_model,
                                         global_para, round, extra_info=extra_info)
            # print(global_model.classifier_1.state_dict())
            global_model.to(device)
            train_acc, avg_time = compute_accuracy(global_model, train_dl_global, device=device, special=True, mode=0, dataset='cifar10')
            test_acc, conf_matrix, avg_time = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True, device=device, special=True, mode=0, dataset='cifar10')
            inference_time += avg_time

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))
            
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            accs[round] = test_acc
            print(round, test_acc)
        print(accs)
        np.save(args.accdir, accs)

    print('all done', args.alg)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"running time: {elapsed_time}s, averaged inference time: {inference_time.total_seconds() * 1_000_000 / args.comm_round}us")
    print(args)
    logger.info(args)