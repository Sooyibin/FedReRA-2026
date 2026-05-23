import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models


class FedAvgNet_header(nn.Module):
    def __init__(self, in_features=3, dim=1600):
        super(FedAvgNet_header, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            # nn.GroupNorm(32, 32, eps=1e-05, affine=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            # nn.GroupNorm(64, 64, eps=1e-05, affine=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x = x.view(x.size()[0], -1)
        x=self.fc1(x)
        return x
    
    def forward_f(self, x):
        outputs = []
        x = self.conv1(x)
        outputs.append(x.view(x.size()[0], -1))
        x = self.conv2(x)
        outputs.append(x.view(x.size()[0], -1))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        outputs.append(x.view(x.size()[0], -1))
        return outputs

class FedAvgNet(nn.Module):
    def __init__(self, args, in_features=3, num_classes=10, dim=1024):
        super().__init__()
        device = args.device
        self.args = args
        print(args.shallow, args.debias_shallow, args.debias_deep)
        dim_dict = {0:6272, 1:1600, 2:512}
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            # nn.GroupNorm(32, 32, eps=1e-05, affine=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            # nn.GroupNorm(64, 64, eps=1e-05, affine=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(dim=1)
        self.bias = torch.zeros((dim_dict[args.layer]), dtype=torch.float).to(device)

        self.classifier_1 = nn.Linear(dim_dict[args.layer], num_classes, bias=False)
        self.classifier_2 = nn.Linear(512, num_classes, bias=False)
        self.global_classifier = nn.Linear(dim, num_classes, bias=False)


    def forward_f(self, x):
        outputs = []
        x = self.conv1(x)

        outputs.append(x.view(x.size()[0], -1))
        x = self.conv2(x)

        outputs.append(x.view(x.size()[0], -1))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)

        outputs.append(x.view(x.size()[0], -1))
        return outputs
    
    def classify(self, f_list, global_classifier=False):
        if(self.args.shallow):
            pred = [self.classifier_1(f_list[self.args.layer]), self.classifier_2(f_list[2])]
        else:
            pred = [self.classifier_2(f_list[2])]
        return pred
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x = x.view(x.size()[0], -1)
        x=self.fc1(x)
        pred = self.classifier_2(x)
        return pred
    
    def forward_method(self, x, mode):
        features = self.forward_f(x)
        features[self.args.layer] = features[self.args.layer] - self.bias
        output = self.classify(features)
        pred = self.softmax(sum([self.softmax(out) for out in output]))
        return pred
    
    def register(self, image):
        f_mean = self.forward_f(image)
        if(self.args.debiased_inf):
            self.bias = torch.mean(f_mean[self.args.layer], dim=0).detach()

class VGG16(nn.Module):
    def __init__(self, args, num_classes=10):
        device = args.device
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv8 = nn.Sequential(
            nn.Conv2d(160, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True))
        self.conv9 = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True))
        self.conv10 = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv11 = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True))
        self.conv12 = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(240, 240, kernel_size=3, padding=1),
            nn.BatchNorm2d(240),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1))
        self.fc1 = nn.Sequential(
            nn.Linear(240, 2048),
            nn.ReLU(True),
            nn.Dropout())
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout())
        self.classifier_1 = nn.Linear(6144, num_classes)
        self.classifier_2 = nn.Linear(1024, num_classes)
        self.bias_1 = torch.zeros(6144, dtype=torch.float).to(device)
        self.bias_2 = torch.zeros(1024, dtype=torch.float).to(device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward_f(self, x):
        features = []
        x = self.conv1(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv2(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv3(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv4(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv5(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv6(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv7(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv8(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv9(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv10(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv11(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv12(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv13(x)
        features.append(x.view(x.size()[0], -1))
        x=x.view(x.size()[0], -1)
        x = self.fc1(x)
        features.append(x.view(x.size()[0], -1))
        x = self.fc2(x)
        features.append(x.view(x.size()[0], -1))
        return features
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x=x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier_2(x)
        return x
    
    def classify(self, f_list):
        pred = [self.classifier_1(f_list[3]), self.classifier_2(f_list[-1])]
        return pred
    
    def forward_method(self, x, mode):
        features = []
        x = self.conv1(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv2(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv3(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv4(x)
        features.append(x.view(x.size()[0], -1) - self.bias_1)
        x = self.conv5(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv6(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv7(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv8(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv9(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv10(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv11(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv12(x)
        features.append(x.view(x.size()[0], -1))
        x = self.conv13(x)
        features.append(x.view(x.size()[0], -1))
        
        x=x.view(x.size()[0], -1)
        x = self.fc1(x)
        features.append(x.view(x.size()[0], -1))
        x = self.fc2(x)
        features.append(x.view(x.size()[0], -1))
        output = self.classify(features)
        pred = self.softmax(sum([self.softmax(out) for out in output]))
        return pred
    
    def register(self, image):
        f_mean = self.forward_f(image)
        self.bias_1 = torch.mean(f_mean[3], dim=0).detach()


"""
class AlexNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(AlexNet, self).__init__()
        device = args.device
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,96, kernel_size=11, stride=4, padding=2),
            nn.GroupNorm(96, 96, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256, kernel_size=5, padding=2),
            nn.GroupNorm(256, 256, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(384, 384, eps=1e-05, affine=True),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(384, 384, eps=1e-05, affine=True),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,128,kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(128, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(21632,2048),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.bias_1 = torch.zeros((21632), dtype=torch.float).to(device)
        self.bias_2 = torch.zeros((1024), dtype=torch.float).to(device)
        self.classifier_1 = nn.Linear(21632, num_classes)
        self.classifier_2 = nn.Linear(1024, num_classes)
    
    def forward_f(self, x):
        features = []
        x=self.conv1(x)
        features.append(x)
        x=self.conv2(x)
        features.append(x)
        x=self.conv3(x)
        features.append(x)
        x=self.conv4(x)
        features.append(x)
        x=self.conv5(x)
        features.append(x)
        x=x.view(x.size()[0], -1)
        x=self.fc1(x)
        features.append(x)
        x=self.fc2(x)
        features.append(x)
        return features
    
    def classify(self, f_list):
        if(self.args.shallow):
            pred = [self.classifier_1(f_list[4]), self.classifier_2(f_list[6])]
        else:
            pred = [self.classifier_2(f_list[6])]
        return pred

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(x.size()[0], -1)
        x=self.fc1(x)
        x=self.fc2(x)
        pred=self.classifier(x)
        return pred
    
    def forward_method(self, x, mode):
        features = []
        
        f1 = self.conv1(x)
        features.append(f1.view(f1.size()[0], -1))
        
        f2 = self.conv2(f1)
        features.append(f2.view(f2.size()[0], -1))
        
        f3 = self.conv3(f2)
        features.append(f3.view(f3.size()[0], -1))

        f4 = self.conv4(f3)
        features.append(f4.view(f4.size()[0], -1))

        f5 = self.conv5(f4)
        features.append(f5.view(f5.size()[0], -1) - self.bias_1)

        f5 = f5.view(f5.size()[0], -1)
        f6 = self.fc1(f5)
        features.append(f6.view(f6.size()[0], -1))

        f7 = self.fc1(f6)
        features.append(f7.view(f7.size()[0], -1) - self.bias_2)
        output = self.classify(features)
        if(mode == 0):
            pred = self.softmax(sum([self.softmax(out) for out in output]))
        else:
            pred = output[0]
        return pred

    def register(self, image):
        f_mean = self.forward_f(image)
        if(self.args.debias_shallow):
            self.bias_1 = torch.mean(f_mean[4], dim=0).detach()
        if(self.args.debias_deep):
            self.bias_2 = torch.mean(f_mean[6], dim=0).detach()
"""

class FedFANet(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.name = name
        
        if self.name == 'cifar10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192) #args.dims_feature=192
            self.classifier = nn.Linear(192, self.n_cls)

            self.bias_1 = torch.zeros((64*5*5), dtype=torch.float).to(args.device)
            self.classifier_1 = nn.Linear(64*5*5, self.n_cls, bias=True)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            y_feature = F.relu(self.fc2(x))
            #y_feature = self.fc2(x)
            x = self.classifier(y_feature)
        return x

    def forward_f(self, x):
        outputs = []
        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            outputs.append(x.view(x.size()[0], -1))
            x = self.pool(F.relu(self.conv2(x)))
            outputs.append(x.view(x.size()[0], -1))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            outputs.append(x.view(x.size()[0], -1))
            x = F.relu(self.fc2(x))
            outputs.append(x.view(x.size()[0], -1))
        return outputs

    def classify(self, f_list):
        if(self.args.shallow):
            pred = [self.classifier_1(f_list[1]), self.classifier(f_list[-1])]
        else:
            pred = [self.classifier(f_list[-1])]
        return pred
    
    def forward_method(self, x, mode):
        if self.name == 'cifar10':
            features = self.forward_f(x)
            features[1] = features[1] - self.bias_1
            # print(features[2].shape)
            output = self.classify(features)
            pred = self.softmax(sum([self.softmax(out) for out in output]))
        return pred
    
    def register(self, image):
        f_mean = self.forward_f(image)
        if(self.args.debiased_inf):
            self.bias_1 = torch.mean(f_mean[1], dim=0).detach()

class fastText(nn.Module):
    def __init__(self, args, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()
        device = args.device
        self.args = args
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output Layer
        self.classifier_2 = nn.Linear(hidden_dim, num_classes)

        self.classifier_1 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.bias_1 = torch.zeros((hidden_dim), dtype=torch.float).to(device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward_f(self, x):
        text, text_lengths = x
        outputs = []
        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        outputs.append(h.view(h.size()[0], -1))
        return outputs

    def classify(self, f_list):
        if(self.args.shallow):
            pred = [self.classifier_1(f_list[0]), self.classifier_2(f_list[-1])]
        else:
            pred = [self.classifier_2(f_list[2])]
        return pred

    def forward_method(self, x, mode):
        text, text_lengths = x
        features = []
        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        features.append(h.view(h.size()[0], -1) - self.bias_1)
        # print(features[2].shape)
        output = self.classify(features)
        pred = self.softmax(sum([self.softmax(out) for out in output]))
        # pred = output[1]
        return pred

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.classifier_2(h)
        out = F.log_softmax(z, dim=1)

        return out
    
    def register(self, image):
        f_mean = self.forward_f(image)
        if(self.args.debiased_inf):
            self.bias_1 = torch.mean(f_mean[self.args.layer], dim=0).detach()

class TextCNN(nn.Module):
    def __init__(self, hidden_dim, num_channels=100, kernel_size=[3,4,5], max_len=200, dropout=0.8, 
                padding_idx=0, vocab_size=98635, num_classes=10):
        super(TextCNN, self).__init__()
        device = args.device
        self.args = args
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        
        # This stackoverflow thread clarifies how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0]+1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1]+1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2]+1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully-Connected Layer
        self.classifier_2 = nn.Linear(num_channels*len(kernel_size), num_classes)

        self.classifier_1 = nn.Linear(hidden_dim, num_classes, bias=False)
        self.bias_1 = torch.zeros((hidden_dim), dtype=torch.float).to(device)
        self.softmax = nn.Softmax(dim=1)
    
    def forward_f(self, x):
        text, text_lengths = x
        outputs = []
        embedded_sent = self.embedding(text)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        outputs.append(conv_out1.view(conv_out1.size()[0], -1))

        

        return outputs

    def forward(self, x):
        text, text_lengths = x

        embedded_sent = self.embedding(text).permute(0,2,1)
        
        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)
        
        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out