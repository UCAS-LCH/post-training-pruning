import torch
import torch.nn as nn
import torch_pruning as tp

import typing
from sklearn.metrics import pairwise_distances

class SimilarityImportance(tp.importance.Importance):
    def __init__(self, group_reduction="mean", normalizer="mean"):
        self.group_reduction = group_reduction
        self.normalizer = normalizer

    def _normalize(self, group_importance, normalizer):
        if normalizer is None: 
            return group_importance
        elif isinstance(normalizer, typing.Callable):
            return normalizer(group_importance)
        elif normalizer == "sum":
            return group_importance / group_importance.sum()
        elif normalizer == "standarization":
            return (group_importance - group_importance.min()) / (group_importance.max() - group_importance.min()+1e-8)
        elif normalizer == "mean":
            return group_importance / group_importance.mean()
        elif normalizer == "max":
            return group_importance / group_importance.max()
        elif normalizer == 'gaussian':
            return (group_importance - group_importance.mean()) / (group_importance.std()+1e-8)
        else:
            raise NotImplementedError

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp

    def get_cosine_distance(self, w, reduction="mean"):
        #cosine, correlation, l1, l2, euclidean
        dis_matrix = pairwise_distances(w.cpu(), metric="l2") 
        #dis_matrix = pairwise_distances(w.cpu(), metric="cosine") 
        dis_matrix = torch.tensor(dis_matrix)
        if reduction=="max":
            #Set diagonal distance to maximum to prevent selection
            max_value = torch.max(dis_matrix)
            inf = torch.ones(dis_matrix.shape[0]) * max_value
            inf = torch.diag_embed(inf)
            dis_matrix = dis_matrix + inf
 
            dis = torch.zeros(dis_matrix.shape[0])
            for i in range(dis_matrix.shape[0]):
                min_value = torch.min(dis_matrix, dim=1)[0]
                min_index = torch.argmin(min_value)
                dis[min_index] = min_value[min_index]
                dis_matrix[min_index] = max_value
                dis_matrix[:,min_index] = max_value
        elif reduction=="sum":
            dis = torch.sum(dis_matrix.abs(), dim=1)
        elif reduction=="mean":
            dis = torch.mean(dis_matrix.abs(), dim=1)
        return dis
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        #Get group norm
        #print(group)
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            # Conv out_channels
            if prune_fn in [
                tp.prune_conv_out_channels,
                tp.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                local_norm = self.get_cosine_distance(w)
                if ch_groups>1:
                    local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                group_imp.append(local_norm)

            elif prune_fn in [
                tp.prune_conv_in_channels,
                tp.prune_linear_in_channels,
            ]:
                is_conv_flatten_linear = False
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)
                if ch_groups > 1 and prune_fn == tp.prune_conv_in_channels and layer.groups == 1:
                    # non-grouped conv and group convs
                    w = w.view(w.shape[0] // group_imp[0].shape[0],
                               group_imp[0].shape[0], w.shape[1]).transpose(0, 1).flatten(1)
                local_norm = self.get_cosine_distance(w)
                if ch_groups > 1:
                    if len(local_norm) == len(group_imp[0]):
                        local_norm = local_norm.view(ch_groups, -1).sum(0)
                    local_norm = local_norm.repeat(ch_groups)
                local_norm = local_norm[idxs]
                group_imp.append(local_norm)

        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
