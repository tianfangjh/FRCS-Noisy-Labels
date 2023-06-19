from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
from autoaugment import CIFAR10Policy, ImageNetPolicy
from torchnet.meter import AUCMeter
import torch.nn.functional as F 
from Asymmetric_Noise import *
from sklearn.metrics import confusion_matrix



def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset): 
    def __init__(self, dataset, sample_ratio, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='',
    clean_idx_=None, noise_idx_=None, recovered_labels=None, pred_idx=[]): 
        
        self.r = r # noise ratio
        self.sample_ratio = sample_ratio
        self.transform = transform
        self.mode = mode
        self.probability = probability
        self.recovered_labels = recovered_labels
        root_dir_save = root_dir
        self.pred_idx = pred_idx

        if dataset == 'cifar10':
            root_dir = './data/cifar10/cifar-10-batches-py'            
            num_class =10         
        else:
            root_dir = './data/cifar100/cifar-100-python'
            num_class =100

        ## For Asymmetric Noise (CIFAR10)    
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} 

        num_sample     = 50000
        self.class_ind = {}
        if self.mode=='test':
            if dataset=='cifar10':    
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                root_dir = './data/cifar100/cifar-100-python'
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    # print(data_dic['data'])
                    train_data.append(data_dic['data'])
                    
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                self.train_data = train_data
                self.train_label = train_label
                if self.recovered_labels is not None:
                    clean_num = [int(train_label[i]==self.recovered_labels[i]) for i in range(50000)]
                    clean_ratio = np.sum(clean_num) / 50000
                    print("clean ratio for warmup: ", clean_ratio)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                self.train_label = train_label

            print(train_data.shape)
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            
            if os.path.exists(noise_file):             
                noise_label = np.load(noise_file)['label']
                noise_idx = np.load(noise_file)['index']
                idx       = list(range(50000))
                clean_idx = [x for x in idx if x not in noise_idx]
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]

            else:       ## Inject Noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                
                if noise_mode == 'asym':
                    if dataset== 'cifar100':
                        noise_label, prob11 =  noisify_cifar100_asymmetric(train_label, self.r)
                    else:
                        for i in range(50000):
                            if i in noise_idx:
                                    noiselabel = self.transition[train_label[i]]
                                    noise_label.append(noiselabel)
                            else:
                                noise_label.append(train_label[i])   
                else:
                    for i in range(50000):
                        if i in noise_idx:
                            if noise_mode=='sym':
                                if dataset=='cifar10': 
                                    noiselabel = random.randint(0,9)
                                elif dataset=='cifar100':    
                                    noiselabel = random.randint(0,99)
                                noise_label.append(noiselabel)

                            elif noise_mode=='pair_flip':  
                                noiselabel = self.pair_flipping[train_label[i]]
                                noise_label.append(noiselabel)   
                    
                        else:
                            noise_label.append(train_label[i])   

                print("Save noisy labels to %s ..."%noise_file)        
                np.savez(noise_file, label = noise_label, index = noise_idx)          
                for kk in range(num_class):
                    self.class_ind[kk] = [i for i,x in enumerate(noise_label) if x==kk]    

            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label

                clean_num = [int(self.noise_label[i]==self.train_label[i]) for i in range(50000)]
                clean_ratio = np.sum(clean_num) / 50000
                print("clean ratio for all data: ", clean_ratio)
            else:
                save_file = 'Clean_index_'+ str(dataset) + '_' +str(noise_mode) +'_' + str(self.r) + '.npz'
                save_file = os.path.join(root_dir_save, save_file)
                pred_idx  = np.zeros(int(self.sample_ratio*num_sample))
                if self.mode == "labeled" and clean_idx_ is None:
                    print("case 1")
                    print("sample ratio: ", self.sample_ratio)
                    pred_idx  = np.zeros(int(self.sample_ratio*num_sample))
                    class_len = int(self.sample_ratio*num_sample/num_class)
                    size_pred = 0
                    print("----------before selecting----------")
                    ## Ranking-based Selection and Introducing Class Balance
                    for i in range(num_class):
                        class_indices = self.class_ind[i]
                        prob1  = np.argsort(probability[class_indices])
                        size1 = len(class_indices)

                        try:
                            pred_idx[size_pred:size_pred+class_len] = np.array(class_indices)[prob1[0:class_len].astype(int)].squeeze()
                            size_pred += class_len
                        except:                            
                            pred_idx[size_pred:size_pred+size1] = np.array(class_indices)
                            size_pred += size1
                    # pred_idx is the index list of clean samples
                    
                    pred_idx = [int(x) for x in list(pred_idx)]
                    pred_idx = pred_idx[0:size_pred]
                    pred_idx = np.array(pred_idx)
                    pred_idx.astype("int64")
                    self.pred_idx = pred_idx
                    probability[probability<0.5] = 0
                    self.probability = [1-probability[i] for i in pred_idx]


                elif self.mode == "unlabeled" and noise_idx_ is None:
                    print("case 2")
                    pred_idx = self.pred_idx
                    idx = list(range(num_sample)) #50000
                    pred_idx_noisy = [int(x) for x in idx if x not in pred_idx]        
                    pred_idx = pred_idx_noisy   
                    pred_idx = np.array(pred_idx)
                    pred_idx.astype("int64")

                if clean_idx_ is not None and noise_idx_ is None:
                    print("case 3")
                    self.train_data = train_data[clean_idx_]
                    self.noise_label = self.recovered_labels[clean_idx_]

                    clean_num = [int(self.recovered_labels[j]==self.train_label[j]) for j in clean_idx_]
                    clean_ratio = np.sum(clean_num) / len(clean_idx_)
                    print("use clean index selecting, labeled data clean ratio: ", clean_ratio)

                    ## Weights for label refinement
                    print("len probability: ", len(self.probability))
                    self.probability = np.array(self.probability)
                    self.probability[self.probability<0.5] = 0
                    self.probability = [1-self.probability[i] for i in clean_idx_]
                    print("clean train data length: ", len(self.train_data))
                elif noise_idx_ is not None:
                    print("case 4")
                    self.train_data = train_data[noise_idx_]
                    self.noise_label = self.recovered_labels[noise_idx_]

                    clean_num = [int(self.recovered_labels[j]==self.train_label[j]) for j in noise_idx_]
                    clean_ratio = np.sum(clean_num) / len(noise_idx_)
                    print("use noise index selecting, unlabeled data clean ratio: ", clean_ratio)

                    print("noise train data length: ", len(self.train_data))
                else:
                    print("case 5")
                    print(type(pred_idx[0]))
                    print("pred_idx:", pred_idx)
                    self.train_data = train_data[pred_idx]
                    if self.recovered_labels is None:
                        self.noise_label = [noise_label[int(i)] for i in pred_idx]
                        clean_num = [int(noise_label[j]==self.train_label[j]) for j in pred_idx]
                        clean_ratio = np.sum(clean_num) / len(pred_idx)
                        print("balanced selecting, clean ratio: ", clean_ratio)
                    else:
                        print("use recovered labels")
                        self.noise_label = [self.recovered_labels[int(i)] for i in pred_idx]
                        clean_num = [int(self.recovered_labels[j]==self.train_label[j]) for j in pred_idx]
                        clean_ratio = np.sum(clean_num) / len(pred_idx)
                        print("balanced selecting, clean ratio: ", clean_ratio)
                print("----------end function--------")

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)

            return img1, img2, img3, img4,  target, prob   

        elif self.mode=='unlabeled':
            img = self.train_data[index]
            image = Image.fromarray(img)
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            img3 = self.transform[2](image)
            img4 = self.transform[3](image)
            return img1, img2, img3, img4

        elif self.mode=='all':
            if self.recovered_labels is not None:
                img, target = self.train_data[index], self.recovered_labels[index]
            else:
                img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index

        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)   
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        
        if self.dataset=='cifar10':
            transform_weak_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            transform_strong_10 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_10,
                "unlabeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
                "labeled": [
                            transform_weak_10,
                            transform_weak_10,
                            transform_strong_10,
                            transform_strong_10
                        ],
            }

            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])

        elif self.dataset=='cifar100':
            transform_weak_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            transform_strong_100 = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )

            self.transforms = {
                "warmup": transform_weak_100,
                "unlabeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
                "labeled": [
                            transform_weak_100,
                            transform_weak_100,
                            transform_strong_100,
                            transform_strong_100
                        ],
            }        
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])

    def run(self, sample_ratio, mode, pred=[], prob=[], clean_idx=None, noise_idx=None, recovered_labels=None):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["warmup"], mode="all",noise_file=self.noise_file, recovered_labels=recovered_labels)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            print("begin constructing labeled dataset")
            labeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["labeled"], mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log, clean_idx_=clean_idx, recovered_labels=recovered_labels)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last=True)  
            print("labeled datasize: ", len(labeled_dataset))
            print("finish constructing labeled dataset")
            print("begin constructing unlabeled dataset")
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["unlabeled"], mode="unlabeled", noise_file=self.noise_file, pred=pred, noise_idx_=noise_idx, recovered_labels=recovered_labels, pred_idx=labeled_dataset.pred_idx)
            clean_ratio = len(labeled_dataset) / 50000
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size= int(self.batch_size/(2*sample_ratio)),
                shuffle=True,
                num_workers=self.num_workers, drop_last =True)    
            print("unlabeled datasize: ", len(unlabeled_dataset))
            print("finish constructing unlabeled dataset")
            return labeled_trainloader, unlabeled_trainloader                
        elif mode=='simclr':
            print("begin constructing unlabeled dataset")
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transforms["unlabeled"], mode="unlabeled", noise_file=self.noise_file, pred=pred, noise_idx_=noise_idx)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size= self.batch_size,
                shuffle=True,
                num_workers=self.num_workers, drop_last =True)    
            print("unlabeled datasize: ", len(unlabeled_dataset))
            print("finish constructing unlabeled dataset")
            return unlabeled_trainloader   

        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, sample_ratio= sample_ratio, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=100,
                shuffle=False,
                num_workers=self.num_workers, drop_last= True)          
            return eval_loader        
