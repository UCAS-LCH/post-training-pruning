import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights, \
    resnet101, ResNet101_Weights, mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.vision_transformer import VisionTransformer

import models
from datasets import *
import torch_pruning as tp
import cosine_imp
from calibration import global_calibration
from sparse_train import sparse_train

parser = argparse.ArgumentParser(description='PyTorch CIFAR post training pruning')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 123)')
parser.add_argument('--model-dir', default='./checkpoints',
                    help='directory of model for saving checkpoint')
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'imagenet'], default="CIFAR10",
                    help="dataset name")
parser.add_argument('--num-samples', type=int, default=512,
                    help="number of samples in the calibration dataset")
parser.add_argument('--model', choices=['ResNet18', 'ResNet50', 'vgg19'], default="ResNet18",
                    help="model name")
parser.add_argument('--imp', choices=['sim', 'l2norm', 'l1norm', 'random', 'lamp', 'group'], 
                    default="l2norm", help="importance metric")
parser.add_argument('--global-pruning', action='store_true', default=False,
                    help='apply global pruning')
parser.add_argument('--iters', type=int, default=1,
                    help='number of iteration steps for pruning')
parser.add_argument('--iters_c', type=int, default=10,
                    help='number of iteration steps for calibration')
parser.add_argument('--iters_s', type=int, default=20,
                    help='number of iteration steps for sparse regularization training')
parser.add_argument('--sparsity', type=float, default=0.5,
                    help='pruning rate')
parser.add_argument('--reg', type=float, default=0.02,
                    help='sparse regularization coefficient')
parser.add_argument('--inc', type=float, default=0.02,
                    help='sparse regularization increment')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
train_imagenet = False
if args.dataset == 'CIFAR10':
    train_loader, test_loader, num_class = cifar10_dataloader(batch_size=args.batch_size, 
        num_samples=args.num_samples)
elif args.dataset == 'CIFAR100':
    train_loader, test_loader, num_class = cifar100_dataloader(batch_size=args.batch_size, 
        num_samples=args.num_samples)
elif args.dataset == 'imagenet':
    train_loader, test_loader, num_class = imagenet_dataloader(batch_size=args.batch_size,
        num_samples=args.num_samples)
    train_imagenet = True

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, test_accuracy

def get_pruner(model, example_inputs, args):
    if args.imp == "sim":
        imp = cosine_imp.SimilarityImportance()
    elif args.imp == "l2norm":
        imp = tp.importance.MagnitudeImportance(p=2)
    elif args.imp == "l1norm":
        imp = tp.importance.MagnitudeImportance(p=1)
    elif args.imp == "lamp":
        imp = tp.importance.LAMPImportance(p=1)
    elif args.imp == "group":
        imp = tp.importance.GroupNormImportance(p=2)
    elif args.imp == "random":
        imp = tp.importance.RandomImportance()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_class:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    round_to = None
    unwrapped_parameters = None
    channel_groups = {}
    if isinstance(model, VisionTransformer):
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                channel_groups[m] = m.num_heads

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=imp,
        iterative_steps=args.iters,
        pruning_ratio=args.sparsity,
        global_pruning=args.global_pruning,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        out_channel_groups=channel_groups,
    )
    return pruner


def main():

    if train_imagenet:
        #model_ori = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        #model_p = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        #model_ori = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
        #model_p = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
        #model_ori = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
        #model_p = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
        model_ori = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).to(device)
        model_p = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).to(device)
        #model_ori = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        #model_p = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    else:
        model_ori = models.__dict__[args.model](num_classes=num_class).to(device)
        model_ori.load_state_dict(torch.load('checkpoints/{}_{}.pt'.format(args.model, args.dataset)))

        model_p = models.__dict__[args.model](num_classes=num_class).to(device)
        model_p.load_state_dict(torch.load('checkpoints/{}_{}.pt'.format(args.model, args.dataset)))
    
    if not train_imagenet:
        example_inputs = torch.randn(1, 3, 32, 32).to(device)
    else:
        example_inputs = torch.randn(1, 3, 224, 224).to(device)
    
    #Get groups of original model
    ignored_layers = []
    ignored_params = []
    for m in model_ori.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_class:
            ignored_layers.append(m) # DO NOT prune the final classifier!

    DG = tp.DependencyGraph().build_dependency(model_ori, example_inputs=example_inputs, ignored_params=ignored_params)
    groups_ori = DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear])
    keep_idxs = {}
    input_idxs = {}
    for group in groups_ori:
        for dep, _ in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_out_channels, \
            tp.prune_multihead_attention_out_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_out_channels ] \
            and dep.target._name not in keep_idxs.keys():
                if hasattr(dep.target.module, 'in_channels'): #Conv
                    input_idxs[dep.target._name] = list(range(dep.target.module.in_channels))
                    keep_idxs[dep.target._name] = list(range(dep.target.module.out_channels))
                elif hasattr(dep.target.module, "num_features"): #BatchNorm
                    input_idxs[dep.target._name] = list(range(dep.target.module.num_features))
                    keep_idxs[dep.target._name] = list(range(dep.target.module.num_features))
                elif hasattr(dep.target.module, "embed_dim"): #MultiheadAttention
                    input_idxs[dep.target._name] = list(range(dep.target.module.embed_dim))
                    keep_idxs[dep.target._name] = list(range(dep.target.module.embed_dim))
                elif hasattr(dep.target.module, "in_features"): #Liner
                    input_idxs[dep.target._name] = list(range(dep.target.module.in_features))
                    keep_idxs[dep.target._name] = list(range(dep.target.module.out_features))
                elif hasattr(dep.target.module, "normalized_shape"): #LayerNorm
                    input_idxs[dep.target._name] = list(range(dep.target.module.normalized_shape[-1]))
                    keep_idxs[dep.target._name] = list(range(dep.target.module.normalized_shape[-1]))

    ignored_layers = []
    for m in model_p.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == num_class:
            ignored_layers.append(m) # DO NOT prune the final classifier!
    DG_p = tp.DependencyGraph().build_dependency(model_p, example_inputs=example_inputs, ignored_params=ignored_params)
    if args.iters > 0:
        pruner = get_pruner(model_p, example_inputs, args)
        sparse_pruner = get_pruner(model_p, example_inputs, args)
    
    base_macs, base_nparams = tp.utils.count_ops_and_params(model_p, example_inputs)
    for i in range(args.iters):
        sparse_train(args, model_ori, model_p, train_loader,
            DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear]),
            DG_p.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear]),
            sparse_pruner, keep_idxs, device)
        global_calibration(args, model_ori, model_p, train_loader,
            DG.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear]),
            DG_p.get_all_groups(ignored_layers=ignored_layers, root_module_types=[nn.Conv2d, nn.Linear]),
            pruner, keep_idxs, device)
        macs, nparams = tp.utils.count_ops_and_params(model_p, example_inputs)
        print("  Iter %d/%d, Params: %.2f M => %.2f M"
              % (i+1, args.iters, base_nparams / 1e6, nparams / 1e6))
        print("  Iter %d/%d, MACs: %.2f G => %.2f G"
              % (i+1, args.iters, base_macs / 1e9, macs / 1e9))
    # evaluation on natural examples
    print('================================================================')
    test_loss, test_acc = eval_test(model_p, device, test_loader)
    print('================================================================')

if __name__ == '__main__':
    main()
