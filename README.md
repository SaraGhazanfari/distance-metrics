# distance-metrics

In this project, we try to calculate the distances in terms of Linf, L2, LPIPS and R-LPIPS (which is a robust version of LPIPS)
between train dataset of three datasets: CIFAR-10, SVHN, Tiny-Imagenet.

After cloning the requirements should be installed. And calculations can be done as mentioned below:
### CIFAR-10

1- Calculate LPIPS for cifar10: python3 cifar10/cifar_lpips_distance.py

2- Calculate R-LPIPS for cifar10: python3 cifar10/cifar_r_lpips_distance.py

### SVHN

1- Calculate LPIPS for svhn: python3 svhn/svhn_lpips_distance.py

2- Calculate R-LPIPS for svhn: python3 svhn/svhn_r_lpips_distance.py

3- Calculte the max distance between classes: python3 svhn/svhn_max_distance.py --path 'path/to/matrix/file'

4- Calculte the min distance between classes: python3 svhn/svhn_min_distance.py --path 'path/to/matrix/file'


