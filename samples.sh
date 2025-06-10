python prune.py --model ResNet50 --dataset imagenet --imp l2norm --iters 1 --global-pruning --sparsity 0.2 --iters_c 10 --reg 0.02 --batch-size 64 --iters_s 20 --num-samples 128
python prune.py --model ResNet50 --dataset imagenet --imp l2norm --iters 1 --global-pruning --sparsity 0.2 --iters_c 10 --reg 0.02 --batch-size 64 --iters_s 20 --num-samples 512
python prune.py --model ResNet50 --dataset imagenet --imp l2norm --iters 1 --global-pruning --sparsity 0.2 --iters_c 10 --reg 0.02 --batch-size 64 --iters_s 20 --num-samples 1024
python prune.py --model ResNet50 --dataset imagenet --imp l2norm --iters 1 --global-pruning --sparsity 0.2 --iters_c 10 --reg 0.02 --batch-size 64 --iters_s 20 --num-samples 2048
python prune.py --model ResNet50 --dataset imagenet --imp l2norm --iters 1 --global-pruning --sparsity 0.2 --iters_c 10 --reg 0.02 --batch-size 64 --iters_s 20 --num-samples 5120

