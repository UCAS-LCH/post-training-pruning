python prune.py --model ResNet50 --dataset CIFAR10 --imp l2norm --iters 1 --global-pruning --sparsity 0.5 --iters_c 10 --reg 0.02 --num-samples 512 --iters_s 20
python prune.py --model ResNet50 --dataset CIFAR100 --imp l2norm --iters 1 --global-pruning --sparsity 0.5 --iters_c 10 --reg 0.02 --num-samples 512 --iters_s 20
python prune.py --model vgg19 --dataset CIFAR10 --imp l2norm --iters 1 --global-pruning --sparsity 0.5 --iters_c 10 --reg 0.02 --num-samples 512 --iters_s 20
python prune.py --model vgg19 --dataset CIFAR100 --imp l2norm --iters 1 --global-pruning --sparsity 0.4 --iters_c 10 --reg 0.02 --num-samples 512 --iters_s 20

