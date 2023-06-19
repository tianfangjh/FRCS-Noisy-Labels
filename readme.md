# FRCS Implementation
This folder is the code implementation of FRCS. We referred to the code implementation of paper "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning".
## Example guidance
### Environment 
- The expeirments are mainly implemented with python3.7, torch1.8.2, torchvision0.9.2

### CIFAR10 experiments
- experiment on cifar10 with 50% symmetric noise
    ```python
    python Train_cifar10.py -data_path ./data/cifar10 --noise_mode 'sym' --r 0.5 --recover_threshold_ratio 0.5 --label_refine --semicon --density_recovery
    ```

### CIFAR100 experiments
- experiment on cifar100 with 50% symmetric noise
    ```python
    python Train_cifar100.py -data_path ./data/cifar100 --noise_mode 'sym' --r 0.5 --recover_threshold_ratio 0.5 --label_refine --semicon --density_recovery
    ```
