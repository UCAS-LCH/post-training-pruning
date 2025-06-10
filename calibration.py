import torch
import torch.nn as nn
import torch_pruning as tp
import numpy as np
from torchvision.models.vision_transformer import VisionTransformer

def global_calibration(args, model_ori, model_p, dataloader, groups_ori, groups_p, pruner, keep_idxs, device):

    def cache_output(name, outputs):
        def tmp(layer, inp, out):
                outputs[name] = out
        return tmp

    output_ori = {}
    handles_ori = []
    visited_ori_layer = {}
    for group in groups_ori:
        for dep, _ in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_out_channels, \
            tp.prune_multihead_attention_out_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_out_channels ] \
            and dep.target._name not in visited_ori_layer.keys():
                handles_ori.append(
                    dep.target.module.register_forward_hook(cache_output(dep.target._name, output_ori))
                )
                visited_ori_layer[dep.target._name] = 1
    
    output_p = {}
    handles_p = []
    visited_p_layer = {}
    for group in groups_p:
        for dep, _ in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_out_channels, \
            tp.prune_multihead_attention_out_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_out_channels] \
            and dep.target._name not in visited_p_layer.keys():
                handles_p.append(
                    dep.target.module.register_forward_hook(cache_output(dep.target._name, output_p))
                )
                visited_p_layer[dep.target._name] = 1


    visited_out = {}
    for group in pruner.step(interactive=True):
        for dep, idx in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_batchnorm_out_channels, tp.prune_multihead_attention_out_channels, tp.prune_layernorm_out_channels] \
            and dep.target._name not in visited_out.keys():
                name = dep.target._name
                visited_out[name] = 1
                keep_idxs[name] = list(set(keep_idxs[name]) - set([keep_idxs[name][j] for j in idx]))
                keep_idxs[name].sort()
        group.prune()

    if isinstance(model_p, VisionTransformer):
        model_p.hidden_dim = model_p.conv_proj.out_channels
    criterion = nn.MSELoss(reduction='sum')
    param_group = []
    num_layers = len(visited_p_layer)
    for name, m in model_p.named_modules():
        if (isinstance(m, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)) \
        or (isinstance(m, nn.BatchNorm2d) and m.affine==True) or (isinstance(m, nn.LayerNorm) and m.elementwise_affine==True)) and name in visited_p_layer.keys():
            param_group.append({'params':m.parameters(), 'lr':args.lr/num_layers})
            num_layers -= 1
    optimizer = torch.optim.Adam(param_group, lr=args.lr)
    #optimizer = torch.optim.Adam(model_p.parameters(), lr=args.lr)

    model_p.train()
    for _ in range(args.iters_c):
        for i, batch in enumerate(dataloader):
            image = batch[0].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                model_ori(image)
            model_p(image)
            loss = 0
            for name in output_p:
                if isinstance(output_p[name],tuple):
                    output_p[name] = output_p[name][0]
                    output_ori[name] = output_ori[name][0]
                if len(output_ori[name].shape) == 4 or len(output_ori[name].shape) == 2:
                    norm = torch.norm(output_ori[name][:,keep_idxs[name]]).item() ** 2
                    loss += criterion(output_p[name], output_ori[name][:,keep_idxs[name]]) / norm
                elif len(output_ori[name].shape) == 3:
                    norm = torch.norm(output_ori[name][:,:,keep_idxs[name]]).item() ** 2
                    loss += criterion(output_p[name], output_ori[name][:,:,keep_idxs[name]]) / norm
            loss.backward()
            optimizer.step()

    for h in handles_p:
        h.remove()
    for h in handles_ori:
        h.remove()


def layer_calibration(args, model_ori, model_p, dataloader, groups_ori, groups_p, pruner, keep_idxs, input_idxs, device):

    def cache_output(name, outputs):
        def tmp(layer, inp, out):
                outputs[name] = out
        return tmp

    def cache_input(name, inputs):
        def tmp(layer, inp, out):
                inputs[name] = inp[0]
        return tmp

    output_ori = {}
    handles_ori = []
    handles_p = []
    input_ori = {}
    visited_ori_layer = {}
    for group in groups_ori:
        for dep, _ in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_out_channels, \
            tp.prune_multihead_attention_out_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_out_channels ] \
            and dep.target._name not in visited_ori_layer.keys():
                handles_ori.append(
                    dep.target.module.register_forward_hook(cache_output(dep.target._name, output_ori))
                )
                handles_p.append(
                    dep.target.module.register_forward_hook(cache_input(dep.target._name, input_ori))
                )
                visited_ori_layer[dep.target._name] = 1

    layers = {}
    for group in groups_p:
        for dep, _ in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_out_channels, \
            tp.prune_multihead_attention_out_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_out_channels] \
            and dep.target._name not in layers.keys():
                name = dep.target._name
                layers[dep.target._name] = dep.target.module
 
    visited_out = {}
    visited_inp = {}
    for group in pruner.step(interactive=True):
        for dep, idx in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_batchnorm_out_channels, tp.prune_multihead_attention_out_channels, tp.prune_layernorm_out_channels] \
            and dep.target._name not in visited_out.keys():
                name = dep.target._name
                visited_out[name] = 1
                keep_idxs[name] = list(set(keep_idxs[name]) - set([keep_idxs[name][j] for j in idx]))
                keep_idxs[name].sort()
            if dep.handler in [tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_batchnorm_in_channels, tp.prune_multihead_attention_in_channels, tp.prune_layernorm_in_channels] \
            and dep.target._name not in visited_inp.keys():
                name = dep.target._name
                visited_inp[name] = 1
                input_idxs[name] = list(set(input_idxs[name]) - set([input_idxs[name][j] for j in idx]))
                input_idxs[name].sort()
        group.prune()

    if isinstance(model_p, VisionTransformer):
        model_p.hidden_dim = model_p.conv_proj.out_channels
    criterion = nn.MSELoss(reduction='sum')

    optimizers = {}
    for name in layers.keys():
        #optimizers[name] = torch.optim.Adam([{'params':layers[name].parameters()}], lr=args.lr)
        pm_list = [m for nm,m in layers[name].named_parameters() if 'weight' in nm]
        optimizers[name] = torch.optim.Adam(pm_list, lr=args.lr) 

    for _ in range(args.iters_c):
        for i, batch in enumerate(dataloader):
            image = batch[0].to(device)
            with torch.no_grad():
                model_ori(image)
            for name in layers.keys():
                optimizers[name].zero_grad()
                with torch.enable_grad():
                    if len(input_ori[name].shape) == 4 or len(input_ori[name].shape) == 2:
                        input_p = input_ori[name][:,input_idxs[name]]
                    elif len(input_ori[name].shape) == 3:
                        input_p = input_ori[name][:,:,input_idxs[name]]
                    if isinstance(layers[name], nn.MultiheadAttention):
                        output = layers[name](input_p, input_p, input_p)
                    else:
                        output = layers[name](input_p)
                if isinstance(output,tuple):
                    output = output[0]
                    output_ori[name] = output_ori[name][0]
                if len(output_ori[name].shape) == 4 or len(output_ori[name].shape) == 2:
                    norm = torch.norm(output_ori[name][:,keep_idxs[name]]).item() ** 2
                    loss = criterion(output, output_ori[name][:,keep_idxs[name]]) / norm
                elif len(output_ori[name].shape) == 3:
                    norm = torch.norm(output_ori[name][:,:,keep_idxs[name]]).item() ** 2
                    loss = criterion(output, output_ori[name][:,:,keep_idxs[name]]) / norm
                loss.backward()
                optimizers[name].step()

    for h in handles_p:
        h.remove()
    for h in handles_ori:
        h.remove()	
