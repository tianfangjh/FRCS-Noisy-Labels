from __future__ import print_function
from operator import mod
import sys
from turtle import pen
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import random
import os
import argparse
import numpy as np
from PreResNet_cifar import *
import dataloader_cifar as dataloader
from math import log2
from Contrastive_loss import *

import collections.abc
from collections.abc import MutableMapping
from produce_tensors_cifar10 import produce_tensors
from correction_cifar import run_knn

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=30, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=0.025, type=float, help='weight for contrastive loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--d_u',  default=0.7, type=float)
parser.add_argument('--tau', default=5, type=float, help='filtering coefficient')
parser.add_argument('--metric', type=str, default = 'JSD', help='Comparison Metric')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool, help = 'Resume from the warmup checkpoint')
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./data/cifar10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_workers', default=6, type=int, help='worker number in dataloader') 
parser.add_argument('--tensor_directory', default='tensor_cifar10', type=str, help='tensor path')
parser.add_argument('--recover_threshold_ratio',  default=0.5, type=float, help= 'recovery threshold ratio for knn')
parser.add_argument('--warm_up',  default=2, type=int, help= 'warm up epochs')
parser.add_argument('--k',  default=50, type=int)
parser.add_argument('--label_refine', action='store_true', help = 'whether label refine for labeled data')
parser.add_argument('--max_sr',  default=0.95, type=float, help= 'maximum select ratio')
parser.add_argument('--density_recovery', action='store_true', help = 'whether recovery labels according to density from sparse to dense')
parser.add_argument('--semicon', action='store_true', help = 'whether use semi contrastive learning')


args = parser.parse_args()
print("hyperparameters: ", args)

## GPU Setup 
torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

## Download the Datasets
torchvision.datasets.CIFAR10(args.data_path,train=True, download=True)
torchvision.datasets.CIFAR10(args.data_path,train=False, download=True)

## Checkpoint Location
folder = args.dataset + '_' + args.noise_mode + args.tensor_directory + '_' + 'with_knn' + '_' + str(args.r) 
model_save_loc = './checkpoint/' + folder
if not os.path.exists(model_save_loc):
    os.mkdir(model_save_loc)

## Log files
stats_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open(model_save_loc +'/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     
test_loss_log = open(model_save_loc +'/test_loss.txt','w')
train_acc = open(model_save_loc +'/train_acc.txt','w')
train_loss = open(model_save_loc +'/train_loss.txt','w')



## For Standard Training 
def warmup_standard(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    
    total_loss = 0.0
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        _, outputs = net(inputs)               
        loss    = CEloss(outputs, labels)
        total_loss += loss.item()  

        if args.noise_mode=='asym':     # Penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        else:   
            L = loss

        L.backward()  
        optimizer.step()                

    sys.stdout.write('\r')
    sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] \t CE-loss: %.4f'
            %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, total_loss / (batch_idx + 1)))
    sys.stdout.flush()

## For Training Accuracy
def warmup_val(epoch,net,optimizer,dataloader):

    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    total = 0
    correct = 0
    loss_x = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
            inputs, labels = inputs.cuda(), labels.cuda() 
            optimizer.zero_grad()
            _, outputs  = net(inputs)               
            _, predicted = torch.max(outputs, 1)    
            loss    = CEloss(outputs, labels)    
            loss_x += loss.item()                      

            total   += labels.size(0)
            correct += predicted.eq(labels).cpu().sum().item()

    acc = 100.*correct/total
    print("\n| Train Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))  
    
    train_loss.write(str(loss_x/(batch_idx+1)))
    train_acc.write(str(acc))
    train_acc.flush()
    train_loss.flush()

    return acc

## Test Accuracy
def test(epoch,net1,net2=None):
    net1.eval()
    if net2 is not None:
        net2.eval()

    num_samples = 1000
    correct = 0
    total = 0
    loss_x = 0
    with torch.no_grad():
        if net2 is not None:
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _, outputs1 = net1(inputs)
                _, outputs2 = net2(inputs)           
                outputs = outputs1+outputs2
                _, predicted = torch.max(outputs, 1)            
                loss = CEloss(outputs, targets)  
                loss_x += loss.item()

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()  
        else:
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                _, outputs = net1(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = CEloss(outputs, targets)  
                loss_x += loss.item()

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item() 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write(str(acc)+'\n')
    test_log.flush()  
    test_loss_log.write(str(loss_x/(batch_idx+1))+'\n')
    test_loss_log.flush()
    return acc


# KL divergence
def kl_divergence(p, q):
    return (p * ((p+1e-10) / (q+1e-10)).log()).sum(dim=1)

## Jensen-Shannon Divergence 
class Jensen_Shannon(nn.Module):
    def __init__(self):
        super(Jensen_Shannon,self).__init__()
        pass
    def forward(self, p,q):
        m = (p+q)/2
        return 0.5*kl_divergence(p, m) + 0.5*kl_divergence(q, m)

## Calculate JSD
def Calculate_JSD(model1, model2, num_samples):  
    JS_dist = Jensen_Shannon()
    JSD   = torch.zeros(num_samples)    

    for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size()[0]

        ## Get outputs of both network
        with torch.no_grad():
            out1 = torch.nn.Softmax(dim=1).cuda()(model1(inputs)[1])     
            out2 = torch.nn.Softmax(dim=1).cuda()(model2(inputs)[1])

        ## Get the Prediction
        out = (out1 + out2)/2     

        ## Divergence clculator to record the diff. between ground truth and output prob. dist.  
        dist = JS_dist(out,  F.one_hot(targets, num_classes = args.num_class))
        JSD[int(batch_idx*batch_size):int((batch_idx+1)*batch_size)] = dist

    return JSD


## Unsupervised Loss coefficient adjustment 
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

num_samples = 50000
warm_up=args.warm_up

## Call the dataloader
loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=args.num_workers,\
    root_dir=model_save_loc,log=stats_log, noise_file='%s/clean_%.4f_%s.npz'%(args.data_path,args.r, args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

# SSL-Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net2.eval() # Freeze one network and train the other
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1

    ## Loss statistics
    loss_x = 0
    loss_u = 0
    loss_scl = 0
    loss_ucl = 0

    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4 = unlabeled_train_iter.next()
        
        batch_size = inputs_x.size(0)

        labeled_labels = labels_x.clone().detach().cuda()
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(), inputs_x4.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda()
        
        with torch.no_grad():
            _, outputs_u11 = net(inputs_u)
            _, outputs_u12 = net(inputs_u2)
            _, outputs_u21 = net2(inputs_u)
            _, outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       

            ptu = pu**(1/args.T)            ## Temparature Sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()                  

            ## Label refinement
            _, outputs_x  = net(inputs_x)
            _, outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            if args.label_refine:
                px = w_x*labels_x + (1-w_x)*px
            else:
                px = labels_x
            ptx = px**(1/args.T)
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()


        simclr_input1 = torch.cat((inputs_x3, inputs_u3), dim=0)
        simclr_input2 = torch.cat((inputs_x4, inputs_u4), dim=0)

        f1, _ = net(simclr_input1)
        f2, _ = net(simclr_input2)
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        unlabeled_labels = np.array([args.num_class + i for i in range(inputs_u4.size(0))])
        unlabeled_labels = torch.from_numpy(unlabeled_labels).cuda()

        all_labels = torch.cat([labeled_labels, unlabeled_labels], dim=0)
        if args.semicon:
            loss_simCLR = contrastive_criterion(features, all_labels)
        else:
            loss_simCLR = contrastive_criterion(features)

        # MixMatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        all_inputs  = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b   = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        ## Mixup
        mixed_input  = l * input_a  + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b

        _, logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
        
        ## Combined Loss
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        ## Regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        ## Total Loss
        loss = Lx + lamb * Lu + args.lambda_c*loss_simCLR + penalty

        loss_x += Lx.item()
        loss_u += Lu.item()
        loss_ucl += loss_simCLR.item()

        # Compute gradient and Do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    sys.stdout.write('\r')
    sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] \t Labeled loss: %.2f  Unlabeled loss: %.2f Contrastive Loss:%.4f'
            %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, loss_x/(batch_idx+1), loss_u/(batch_idx+1),  loss_ucl/(batch_idx+1)))
    sys.stdout.flush()


## Semi-Supervised Loss
criterion  = SemiLoss()

## Optimizer and Scheduler
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, 280, 2e-4)
scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, 280, 2e-4)

## Loss Functions
CE       = nn.CrossEntropyLoss(reduction='none')
CEloss   = nn.CrossEntropyLoss()
MSE_loss = nn.MSELoss(reduction= 'none')
contrastive_criterion = SupConLoss()

if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

## Resume from the warmup checkpoint 
model_name_1 = 'Net1_warmup.pth'
model_name_2 = 'Net2_warmup.pth'

old_epoch = -1

if args.resume:
    net1.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_1))['net'])
    net2.load_state_dict(torch.load(os.path.join(model_save_loc, model_name_2))['net'])

    ckpt1 = torch.load(os.path.join(model_save_loc, "Net1.pth"))
    ckpt2 = torch.load(os.path.join(model_save_loc, "Net2.pth"))
    old_epoch = ckpt1['epoch']



best_acc = 0

## Warmup and SSL-Training 
clean_idx = []
noise_idx = []
weight = []

warmup_trainloader = loader.run(0, 'warmup')

for epoch in range(old_epoch+1,args.num_epochs+1):   
    test_loader = loader.run(0, 'test')
    eval_loader = loader.run(0, 'eval_train')   

    ## Warmup Stage 
    if epoch<warm_up:
        print('Warmup Model')
        warmup_standard(epoch, net1, optimizer1, warmup_trainloader)   

        print('\nWarmup Model')
        warmup_standard(epoch, net2, optimizer2, warmup_trainloader) 
    
    else:
        projections, labels = produce_tensors(tensor_directory=args.tensor_directory, model=net1)

        clean_idx, noise_idx, weights1, recovered_labels1, densities, select_ratio = run_knn(False, False, tensor_directory=args.tensor_directory, noise=args.r, recover_threshold_ratio=args.recover_threshold_ratio, noise_mode=args.noise_mode, k=args.k, num_class=args.num_class, projections=projections, labels=labels, recovery_method=args.density_recovery)
        threshold1 = np.mean(weights1)
        if threshold1>args.d_u:
            threshold1 = threshold1 - (threshold1-np.min(weights1))/args.tau
        SR1 = np.sum(weights1<threshold1)/num_samples
        print("computed sample ratio: ", SR1)
        SR1 = np.clip(SR1, 0, args.max_sr)
        print("cliped computed sample ratio: ", SR1)


        projections, labels = produce_tensors(tensor_directory=args.tensor_directory, model=net2)
        clean_idx, noise_idx, weights2, recovered_labels2, densities, select_ratio = run_knn(False, False, tensor_directory=args.tensor_directory, noise=args.r, recover_threshold_ratio=args.recover_threshold_ratio, noise_mode=args.noise_mode, k=args.k, num_class=args.num_class, projections=projections, labels=labels,recovery_method=args.density_recovery)
        threshold2 = np.mean(weights2)
        if threshold2>args.d_u:
            threshold2 = threshold2 - (threshold2-np.min(weights2))/args.tau

        SR2 = np.sum(weights2<threshold2)/num_samples
        print("computed sample ratio: ", SR2)
        SR2 = np.clip(SR2, 0, args.max_sr)
        print("cliped computed sample ratio: ", SR2)

        print('Train Net1\n')
        print("weights2: ", len(weights2))
        labeled_trainloader, unlabeled_trainloader = loader.run(SR2, 'train', prob=weights2, recovered_labels=recovered_labels2) # Uniform Selection
        train(epoch,net1,net2,optimizer1,labeled_trainloader, unlabeled_trainloader)    # train net1  

        print("weights1: ", len(weights1))
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(SR1, 'train', prob=weights1, recovered_labels=recovered_labels1)     # Uniform Selection
        train(epoch, net2,net1,optimizer2,labeled_trainloader, unlabeled_trainloader)       # train net1

    acc = test(epoch,net1,net2)
    # acc = test(epoch,net1)
    scheduler1.step()
    scheduler2.step()

    if acc > best_acc:
        if epoch <warm_up:
            model_name_1 = 'Net1_warmup.pth'
            model_name_2 = 'Net2_warmup.pth'
        else:
            model_name_1 = 'Net1.pth'
            model_name_2 = 'Net2.pth'            

        print("Save the Model-----")
        checkpoint1 = {
            'net': net1.state_dict(),
            'Model_number': 1,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': optimizer1.state_dict(),
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        checkpoint2 = {
            'net': net2.state_dict(),
            'Model_number': 2,
            'Noise_Ratio': args.r,
            'Loss Function': 'CrossEntropyLoss',
            'Optimizer': optimizer2.state_dict(),
            'Noise_mode': args.noise_mode,
            'Accuracy': acc,
            'Pytorch version': '1.4.0',
            'Dataset': 'TinyImageNet',
            'Batch Size': args.batch_size,
            'epoch': epoch,
        }

        torch.save(checkpoint1, os.path.join(model_save_loc, model_name_1))
        torch.save(checkpoint2, os.path.join(model_save_loc, model_name_2))
        best_acc = acc
    print("best accuracy: ", best_acc)
print("best accuracy: ", best_acc)

