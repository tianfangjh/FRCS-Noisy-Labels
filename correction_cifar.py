
from concurrent.futures import thread
from numpy.ma.core import count
import torch
import numpy as np
import copy

from collections import Counter
import torch.nn as nn
import os
import torch.nn.functional as F

def loadData(use_recoveryed_label=False, tensor_directory="tensor_cifar100", noise=0.8):
    labels = np.array([])
    projections = np.array([[]])
    images = np.array([[[[]]]])
    for i in range(10):
        if i == 0:
            projections = torch.load('./{}/projections{}.pt'.format(tensor_directory, i))
            labels = torch.load('./{}/labels{}.pt'.format(tensor_directory, i))
        else:
            projections = np.vstack((projections, torch.load('./{}/projections{}.pt'.format(tensor_directory, i))))
            labels = np.hstack((labels, torch.load('./{}/labels{}.pt'.format(tensor_directory, i))))
    origin_labels = copy.deepcopy(labels)
    if use_recoveryed_label:
        labels = torch.load("./data/recovered_labels_noise0{}.pt".format(int(noise*10)))

    return labels, projections, images, origin_labels

def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix

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


def run_knn(use_recoveryed_label, evaluation, tensor_directory, noise, recover_threshold_ratio, noise_mode, k, num_class, projections, labels,recovery_method=True):
    print("recovery threshold ratio: ", recover_threshold_ratio)
    useThreshold = True

    origin_labels = copy.deepcopy(labels)
    querys = copy.deepcopy(projections)
    print("projections: ", projections)
    print(projections.shape)
    dimension = 128
    import faiss          
    index2 = faiss.IndexFlatL2(dimension)
    index2.add(projections)
    print(index2.ntotal)
    D2, I2 = index2.search(querys, k)
    # import numpy as np
    densities = 1 / (np.sum(D2, axis=1) / D2.shape[1])
    densities = np.array(densities)

    density_order_idx  = np.argsort(densities)
    print("densities shape: ", densities.shape)
    print("densities: ", densities)
    print("max density: ", np.max(densities))
    print("min density: ", np.min(densities))
    print("mean density: ", np.mean(densities))
    
    stored_noise_label = np.array([])
    if num_class == 100:
        noise_label_file = '%s/clean_%.4f_%s.npz'%('./data/cifar100',noise, noise_mode)
    else:
        noise_label_file = '%s/clean_%.4f_%s.npz'%('./data/cifar10',noise, noise_mode)
    if os.path.exists(noise_label_file):
        stored_noise_label = np.load(noise_label_file)['label']
        
    # import numpy as np
    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    corruption_matrix = corruption_list['uniform'](noise, num_class)

    noise_labels = copy.deepcopy(labels)
    isClean = np.array([1 for i in range(len(labels))])
    if not os.path.exists(noise_label_file):
        for idx in range(len(labels)):
            if not use_recoveryed_label:
                p = corruption_matrix[labels[idx]]
                noise_labels[idx] = np.random.choice(num_class, p=p)
            isClean[idx] = int((noise_labels[idx] == origin_labels[idx]))
    else:
        print("using existing noise label file: ", noise_label_file)
        
        print("origin labels in knn: ", origin_labels)
        for idx in range(len(labels)):
            if not use_recoveryed_label:
                noise_labels[idx] = stored_noise_label[idx]
            isClean[idx] = int((noise_labels[idx] == origin_labels[idx]))

    print("clean labels: {}, total samples: {}, clean ratio: {:.3}".format(sum(isClean), index2.ntotal, sum(isClean) / index2.ntotal))

    # import numpy as np
    from scipy.special import entr
    from scipy.stats import mode
    isCleanRevised_WithoutEntropy = copy.deepcopy(isClean)
    recovered_labels = copy.deepcopy(noise_labels)


    k_neighbors_labels = noise_labels[I2]
    count_time_confidences = []
    label_onehots = F.one_hot(torch.tensor(recovered_labels), num_class)
    pred_onehots = np.zeros((len(density_order_idx), num_class))
    recovery_order = []
    if recovery_method:
        print("using density recovery")
        recovery_order = copy.deepcopy(density_order_idx)
    else:
        print("using random recovery")
        recovery_order = [j for j in range(len(densities))]
    
    for i in recovery_order:
        neighbor_idx = I2[i]
        k_neighbors_labels = recovered_labels[neighbor_idx]
        labels_counter =  dict(Counter(k_neighbors_labels))
        for j in range(num_class):
            if j not in labels_counter.keys():
                pred_onehots[i][j] = 0.0
            else:
                pred_onehots[i][j] = labels_counter[j] / k

        count_time_confidence = mode(k_neighbors_labels).count
        count_time_confidences.append(count_time_confidence)
        revised_label = mode(k_neighbors_labels).mode
        recover_threshold = k * recover_threshold_ratio
        whether_modify = count_time_confidence >= recover_threshold
        if whether_modify:
            recovered_labels[i] = revised_label
    JS_dist = Jensen_Shannon()
    pred_onehots = torch.tensor(pred_onehots).cuda()
    label_onehots = label_onehots.cuda()
    dist = JS_dist(pred_onehots, label_onehots)
    print('JSD shape: ', dist)
    print("JSD: ", dist)
    print("max JSD: ", torch.max(dist))
    print("min JSD: ", torch.min(dist))
    print("mean JSD: ", torch.mean(dist))
    dist_np = dist.cpu().numpy()

    #the number of occurrences of major label
    count_time_confidences = np.squeeze(count_time_confidences, 1)
    print("max value: ", np.max(count_time_confidences))
    print("avg value: ", np.mean(count_time_confidences))
    print("min value: ", np.min(count_time_confidences))
    weight = dist_np

    k_confidence = 50
    D3, I3 = index2.search(querys, k_confidence)
    k_neighbors_labels_confidence = recovered_labels[I3]

    count_time_confidence = mode(k_neighbors_labels_confidence, 1).count
    count_time_confidence = np.squeeze(count_time_confidence, 1)

    #save recovered labels and total images
    print(count_time_confidence)
    mean_time = np.mean(count_time_confidence)
    max_time = np.max(count_time_confidence)
    min_time = np.min(count_time_confidence)
    step = (mean_time - min_time) / 10
    threshold = k_confidence

    greater_than_avg_bool = count_time_confidence >= threshold
    greater_than_avg = [int(i) for i in greater_than_avg_bool]

    clean_idx = np.nonzero(greater_than_avg)[0]
    clean_labels =  Counter(recovered_labels[clean_idx])

    all_idx = np.arange(len(count_time_confidence))
    noise_idx = np.setdiff1d(all_idx, clean_idx)

    isCleanRevised_WithoutEntropy = [int(i) for i in (recovered_labels == origin_labels)]
    clean_sample_number = np.sum(isCleanRevised_WithoutEntropy)

    print("recovered clean labels without entropy: {}, total samples: {}, clean ratio: {:.3}".format(clean_sample_number, index2.ntotal, clean_sample_number / index2.ntotal))

    import sklearn.metrics as sm

    matrixes_withoutentropy = sm.confusion_matrix(isClean, isCleanRevised_WithoutEntropy)

    # print(matrixes_withentropy)
    print("---------recovery confusion matrix----------")
    print(matrixes_withoutentropy)

    print("--------recovery result with threshold----------")
    clean_with_occur = sm.confusion_matrix(isCleanRevised_WithoutEntropy, greater_than_avg)
    print(clean_with_occur)
    select_ratio = np.sum(clean_with_occur[:,1]) / (np.sum(clean_with_occur[:,0]) + np.sum(clean_with_occur[:,1]))
    return clean_idx, noise_idx, weight, recovered_labels, densities, select_ratio

def main():
    run_knn(False, True, tensor_directory="tensor_cifar100", noise=0.5, recover_threshold_ratio=0.3, noise_mode="sym", k=50, num_class=100)

if __name__ == '__main__':
    main()