import torch
import torch.nn as nn
import torch_pruning as tp
import copy

def sparse_train(args, model_ori, model_p, dataloader, groups_ori, groups_p, pruner, keep_idxs, device):

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
    visited_inp = {}
    out_pruned = {}
    out_remained = {}
    input_pruned = {}
    for group in pruner.step(interactive=True):
        for dep, idx in group:
            if dep.handler in [tp.prune_conv_out_channels, tp.prune_linear_out_channels, tp.prune_batchnorm_out_channels, tp.prune_multihead_attention_out_channels, tp.prune_layernorm_out_channels] \
            and dep.target._name not in visited_out.keys():
                name = dep.target._name
                visited_out[name] = 1
                out_pruned[name] = idx
                out_remained[name] = list(set(keep_idxs[name]) - set([keep_idxs[name][j] for j in idx]))
            if dep.handler in [tp.prune_conv_in_channels, tp.prune_linear_in_channels, tp.prune_multihead_attention_in_channels] \
            and dep.target._name not in visited_inp.keys():
                name = dep.target._name
                visited_inp[name] = 1
                input_pruned[name] = idx

    for akey in visited_p_layer.keys():
        if akey not in out_pruned:
            out_pruned[akey] = []
        if akey not in out_remained:
            out_remained[akey] = []
        if akey not in input_pruned:
            input_pruned[akey] = []
    
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model_p.parameters(), lr=args.lr)

    model_p.train()
    for j in range(args.iters_s):
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
                    #norm = torch.norm(output_ori[name][:,out_remained[name]]).item() ** 2
                    #loss += criterion(output_p[name][:,out_remained[name]], output_ori[name][:,out_remained[name]]) / norm
                    norm = torch.norm(output_ori[name][:,keep_idxs[name]]).item() ** 2
                    loss += criterion(output_p[name], output_ori[name][:,keep_idxs[name]]) / norm
                elif len(output_ori[name].shape) == 3:
                    #norm = torch.norm(output_ori[name][:,:,out_remained[name]]).item() ** 2
                    #loss += criterion(output_p[name][:,:,out_remained[name]], output_ori[name][:,:,out_remained[name]]) / norm
                    norm = torch.norm(output_ori[name][:,:,keep_idxs[name]]).item() ** 2
                    loss += criterion(output_p[name], output_ori[name][:,:,keep_idxs[name]]) / norm
            loss.backward()
            for name, m in model_p.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)) and name in visited_p_layer.keys():
                    m.weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.weight.data[out_pruned[name]] * args.lr
                    m.weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.weight.data[:,input_pruned[name]] * args.lr
                elif isinstance(m, (nn.MultiheadAttention)) and name in visited_p_layer.keys():
                    if m.q_proj_weight is not None:
                        m.q_proj_weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.q_proj_weight.data[out_pruned[name]] * args.lr
                        m.q_proj_weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.q_proj_weight.data[:,input_pruned[name]] * args.lr
                        m.k_proj_weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.k_proj_weight.data[out_pruned[name]] * args.lr
                        m.k_proj_weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.k_proj_weight.data[:,input_pruned[name]] * args.lr
                        m.v_proj_weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.v_proj_weight.data[out_pruned[name]] * args.lr
                        m.v_proj_weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.v_proj_weight.data[:,input_pruned[name]] * args.lr
                    else:
                        idxs = out_pruned[name]
                        embed_dim = m.in_proj_weight.shape[1]
                        pruning_idxs_repeated = idxs + [i+ embed_dim for i in idxs] + [i+2*embed_dim for i in idxs]
                        m.in_proj_weight.grad.data[pruning_idxs_repeated] = (args.reg + j * args.inc) * m.in_proj_weight.data[pruning_idxs_repeated] * args.lr
                        m.in_proj_weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.in_proj_weight.data[:,input_pruned[name]] * args.lr
                    m.out_proj.weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.out_proj.weight.data[out_pruned[name]] * args.lr
                    m.out_proj.weight.grad.data[:,input_pruned[name]] = (args.reg + j * args.inc) * m.out_proj.weight.data[:,input_pruned[name]] * args.lr
                elif isinstance(m, (nn.BatchNorm2d)) and m.affine==True and name in visited_p_layer.keys():
                    m.weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.weight.data[out_pruned[name]] * args.lr
                elif isinstance(m, (nn.LayerNorm)) and m.elementwise_affine==True and name in visited_p_layer.keys():
                    m.weight.grad.data[out_pruned[name]] = (args.reg + j * args.inc) * m.weight.data[out_pruned[name]] * args.lr
            optimizer.step()

    model_p.eval()
    for h in handles_p:
        h.remove()
    for h in handles_ori:
        h.remove()
