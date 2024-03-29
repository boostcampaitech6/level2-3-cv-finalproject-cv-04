"""
PET model and criterion classes
"""
import time

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .backbones import *
from .transformer import *
from .position_encoding import build_position_encoding


class BasePETCount(nn.Module):
    """ 
    Base PET model
    """
    def __init__(self, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        hidden_dim = args.hidden_dim

        # prediction head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # classifier
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)  # regressor

        self.pq_stride = args.sparse_stride if quadtree_layer == 'sparse' else args.dense_stride
        self.feat_name = '8x' if quadtree_layer == 'sparse' else '4x'
        self.feats_norm = nn.LayerNorm(hidden_dim)
        
    
    def get_point_query(self, samples, features, encode_feats, **kwargs):
        """
        Generate point query
        """
        src, _ = features[self.feat_name].decompose()

        # get image shape
        input = samples.tensors
        image_shape = torch.tensor(input.shape[2:])
        shape = (image_shape + self.pq_stride//2 - 1) // self.pq_stride
        
        # generate points queries
        shift_x = ((torch.arange(0, shape[1]) + 0.5) * self.pq_stride).long()
        shift_y = ((torch.arange(0, shape[0]) + 0.5) * self.pq_stride).long()
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1, 0) # 2xN --> Nx2
        h, w = shift_x.shape

        # get points queries features, equivalent to nearest interpolation
        shift_y_down, shift_x_down = points_queries[:, 0] // self.pq_stride, points_queries[:, 1] // self.pq_stride  # 왼쪽 위 포인트
        src_feats = src[:, :, shift_y_down, shift_x_down]  # [B C (H W)]
        bs, c = src_feats.shape[:2]
        
        # residual connection
        encode_feats = F.interpolate(encode_feats, (h, w)).reshape(bs, c, -1)
        src_feats = src_feats + encode_feats
        
        win_size_w, win_size_h = kwargs["window_size"]
        if "test" in kwargs:
            # dynamic point query generation
            div = kwargs['div']  # [B, H, W]
            div_win = window_partition(div.unsqueeze(1), win_size_h, win_size_w)  # [B, 1, H, W] -> [L, B_, 1]
            valid_div = (div_win > 0.5).sum(dim=0)[:,0]
            v_idx = valid_div > 0
            
            # window-rize
            src_feats = rearrange(src_feats, "B C (H W) -> B C H W", H=h, W=w)
            points_queries = rearrange(points_queries.unsqueeze(0), "B (H W) N -> B N H W", H=h, W=w)
            
            src_feats = window_partition(src_feats, win_size_h, win_size_w)  # [L B' C]
            points_queries = window_partition(points_queries, win_size_h, win_size_w)  # [L B' 2]
            src_feats = src_feats[:, v_idx].reshape(-1, c)  # [L v_idx C] -> [(L v_idx) C]
            
            v_idx = v_idx.cpu()
            points_queries = points_queries[:, v_idx].reshape(-1, 2)  # [L v_idx 2] -> [(L v_idx) 2]
        else:
            src_feats = src_feats.transpose(1, 2)  # [B (H W) C]
        
        # points_queries = train: [(H W), 2] / test: [(L v_idx) 2]
        # src_feats = train: [B (H W) C] / test: [(L v_idx) C]
        out = (points_queries, src_feats)
        return out
    
    
    def predict(self, samples, points_queries, src_feats, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(src_feats)  # [8, 1024, 2] or [8, 4096, 2]
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(src_feats).sigmoid() - 0.5) * 2.0  # [8, 1024, 2] or [8, 4096, 2]

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]  # train: [256, 256] / test: [H, W]
        img_h, img_w = img_shape  # train: 256, 256 / test: H, W
        
        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h  # shift_y
        points_queries[:, 1] /= img_w  # shift_x

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)

        outputs_points = outputs_offsets + points_queries
        out = {'pred_logits': outputs_class, 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, encode_feats, **kwargs):
        # samples: 원본 이미지(H x W), features: FPN(64x64, 32x32)
        
        # get points queries for prediction
        points_queries, src_feats = self.get_point_query(samples, features, encode_feats, **kwargs)
        src_feats = self.feats_norm(src_feats)
        
        # prediction
        outputs = self.predict(samples, points_queries, src_feats, **kwargs)
        return outputs
    

class PET(nn.Module):
    """ 
    Point quEry Transformerdfdf
    """
    def __init__(self, backbone, num_classes, args=None):
        super().__init__()
        self.backbone = backbone
        
        # positional embedding
        self.pos_embed = build_position_encoding(args)

        # feature projection
        hidden_dim = args.hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1),
            ]
        )

        # context encoder
        self.encode_feats = '8x'
        self.transformer = build_encoder(args)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = \
            context_patch[0]//int(self.encode_feats[:-1]),\
            context_patch[1]//int(self.encode_feats[:-1])  # context_w = 128//8 = 16, context_h = 64//8 = 8
        self.quadtree_splitter = nn.Sequential(  # 32x32
            nn.AvgPool2d(kernel_size=(context_h, context_w), stride=(context_h, context_w)),  # kernel_size = (8, 16), stride = (8, 16)
            # 2x4 -> [B, C, 4, 2]
            nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1),  # [8, 1, 4, 2]
            nn.Sigmoid(),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4  # point-query stride
        self.quadtree_sparse = BasePETCount(num_classes, quadtree_layer='sparse', args=args)
        self.quadtree_dense = BasePETCount(num_classes, quadtree_layer='dense', args=args)

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output_sparse, output_dense = outputs['sparse'], outputs['dense']
        weight_dict = criterion.weight_dict
        warmup_ep = 5

        # compute loss
        if epoch >= warmup_ep:
            loss_dict_sparse = criterion(output_sparse, targets, div=outputs['split_map_sparse'])
            loss_dict_dense = criterion(output_dense, targets, div=outputs['split_map_dense'])
        else:
            loss_dict_sparse = criterion(output_sparse, targets)
            loss_dict_dense = criterion(output_dense, targets)

        # sparse point queries loss
        loss_dict_sparse = {k+'_sp':v for k, v in loss_dict_sparse.items()}
        weight_dict_sparse = {k+'_sp':v for k,v in weight_dict.items()}
        loss_pq_sparse = sum(loss_dict_sparse[k] * weight_dict_sparse[k] for k in loss_dict_sparse.keys() if k in weight_dict_sparse)

        # dense point queries loss
        loss_dict_dense = {k+'_ds':v for k, v in loss_dict_dense.items()}
        weight_dict_dense = {k+'_ds':v for k,v in weight_dict.items()}
        loss_pq_dense = sum(loss_dict_dense[k] * weight_dict_dense[k] for k in loss_dict_dense.keys() if k in weight_dict_dense)
    
        # point queries loss
        losses = loss_pq_sparse + loss_pq_dense 

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_sparse)
        loss_dict.update(loss_dict_dense)

        weight_dict = dict()
        weight_dict.update(weight_dict_sparse)
        weight_dict.update(weight_dict_dense)

        # quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.quadtree_sparse.pq_stride   # dense regions index
        ds_div = outputs['split_map_raw'][ds_idx]
        sp_div = 1 - outputs['split_map_raw']

        # constrain sparse regions
        loss_split_sp = 1 - sp_div.view(bs, -1).max(dim=1)[0].mean()

        # constrain dense regions
        if sum(ds_idx) > 0:
            ds_num = ds_div.shape[0]
            loss_split_ds = 1 - ds_div.view(ds_num, -1).max(dim=1)[0].mean()
        else:
            loss_split_ds = outputs['split_map_raw'].sum() * 0.0

        # update quadtree splitter loss            
        loss_split = loss_split_sp + loss_split_ds
        weight_split = 0.1 if epoch >= warmup_ep else 0.0
        loss_dict['loss_split'] = loss_split
        weight_dict['loss_split'] = weight_split

        # final loss
        losses += loss_split * weight_split
        return {'loss_dict':loss_dict, 'weight_dict':weight_dict, 'losses':losses}

    def forward(self, samples: NestedTensor, **kwargs):
        """
        The forward expects a NestedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        # backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)

        # positional embedding
        dense_input_embed = self.pos_embed(samples)
        kwargs['dense_input_embed'] = dense_input_embed  # [B, 256, 256, 256]

        # feature projection
        features['4x'] = NestedTensor(self.input_proj[0](features['4x'].tensors), features['4x'].mask)
        features['8x'] = NestedTensor(self.input_proj[1](features['8x'].tensors), features['8x'].mask)

        # forward
        if 'train' in kwargs:
            out = self.train_forward(samples, features, pos, **kwargs)
        else:
            out = self.test_forward(samples, features, pos, **kwargs)
        return out

    def pet_forward(self, samples, features, pos, **kwargs):
        # context encoding
        src, _ = features[self.encode_feats].decompose()  # NestedTensor
        src_pos = pos[self.encode_feats]
        
        encode_feats = self.transformer(src, src_pos)  # [B C H W]
        
        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape  # bs = 8, src_h = 32, src_w = 32
        sp_h, sp_w = src_h, src_w  # sp_h = 32, sp_w = 32
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)  # ds_h = 64, ds_w = 64
        split_map = self.quadtree_splitter(encode_feats)  # [8, 1, 4, 2]
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)  # interpolate(split_map) = [8, 1, 64, 64] / interpolate(split_map).reshape = [8, 4096]
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)  # interpolate(split_map) = [8, 1, 32, 32] / interpolate(split_map).reshape = [8, 1024]
        
        # quadtree layer0 forward (sparse)
        if 'train' in kwargs or (split_map_sparse > 0.5).sum() > 0:
            kwargs['div'] = split_map_sparse.reshape(bs, sp_h, sp_w)
            kwargs["window_size"] = [16, 8]
            outputs_sparse = self.quadtree_sparse(samples, features, encode_feats, **kwargs)  # samples: 원본 이미지(HxW), features: FPN(64x64, 32x32), context_info: 인코더 출력(32x32) 
        else:
            outputs_sparse = None
        
        # quadtree layer1 forward (dense)
        if 'train' in kwargs or (split_map_dense > 0.5).sum() > 0:
            kwargs['div'] = split_map_dense.reshape(bs, ds_h, ds_w)
            kwargs["window_size"] = [8, 4]
            outputs_dense = self.quadtree_dense(samples, features, encode_feats, **kwargs)  # samples: 원본 이미지(HxW), features: FPN(64x64, 32x32), context_info: 인코더 출력(32x32) 
        else:
            outputs_dense = None
            
        # format outputs
        outputs = dict()
        outputs['sparse'] = outputs_sparse
        outputs['dense'] = outputs_dense
        outputs['split_map_raw'] = split_map
        outputs['split_map_sparse'] = split_map_sparse
        outputs['split_map_dense'] = split_map_dense
        return outputs
    
    def train_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)

        # compute loss
        criterion, targets, epoch = kwargs['criterion'], kwargs['targets'], kwargs['epoch']
        losses = self.compute_loss(outputs, criterion, targets, epoch, samples)
        return losses
    
    def test_forward(self, samples, features, pos, **kwargs):
        outputs = self.pet_forward(samples, features, pos, **kwargs)
        out_dense, out_sparse = outputs['dense'], outputs['sparse']
        thrs = 0.5  # inference threshold
        
        # process sparse point queries
        if outputs['sparse'] is not None:
            out_sparse_scores = torch.nn.functional.softmax(out_sparse['pred_logits'], -1)[..., 1]
            valid_sparse = out_sparse_scores > thrs
            index_sparse = valid_sparse.cpu()
        else:
            index_sparse = None

        # process dense point queries
        if outputs['dense'] is not None:
            out_dense_scores = torch.nn.functional.softmax(out_dense['pred_logits'], -1)[..., 1]
            valid_dense = out_dense_scores > thrs
            index_dense = valid_dense.cpu()
        else:
            index_dense = None

        # format output
        div_out = dict()
        output_names = out_sparse.keys() if out_sparse is not None else out_dense.keys()
        for name in list(output_names):
            if 'pred' in name:
                if index_dense is None:
                    div_out[name] = out_sparse[name][index_sparse].unsqueeze(0)
                elif index_sparse is None:
                    div_out[name] = out_dense[name][index_dense].unsqueeze(0)
                else:
                    div_out[name] = torch.cat([out_sparse[name][index_sparse].unsqueeze(0), out_dense[name][index_dense].unsqueeze(0)], dim=1)
            else:
                div_out[name] = out_sparse[name] if out_sparse is not None else out_dense[name]
        
        div_out['split_map_raw'] = outputs['split_map_raw']
        return div_out


class SetCriterion(nn.Module):
    """ Compute the loss for PET:
        1) compute hungarian assignment between ground truth points and the outputs of the model
        2) supervise each pair of matched ground-truth / prediction and split map
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """
        Parameters:
            num_classes: one-class in crowd counting
            matcher: module able to compute a matching between targets and point queries
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef    # coefficient for non-object background points
        self.register_buffer('empty_weight', empty_weight)
        self.div_thrs_dict = {8: 0.0, 4:0.5}
    
    def loss_labels(self, outputs, targets, indices, num_points, log=True, **kwargs):
        """
        Classification loss:
            - targets dicts must contain the key "labels" containing a tensor of dim [nb_target_points]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape[:2], dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # compute classification loss
        if 'div' in kwargs:
            # get sparse / dense image index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            ds_idx = den_sort[:len(den_sort)//2]
            sp_idx = den_sort[len(den_sort)//2:]
            eps = 1e-5

            # raw cross-entropy loss
            weights = target_classes.clone().float()
            weights[weights==0] = self.empty_weight[0]
            weights[weights==1] = self.empty_weight[1]
            raw_ce_loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, ignore_index=-1, reduction='none')

            # binarize split map
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs

            # dual supervision for sparse/dense images
            loss_ce_sp = (raw_ce_loss * weights * div_mask)[sp_idx].sum() / ((weights * div_mask)[sp_idx].sum() + eps)
            loss_ce_ds = (raw_ce_loss * weights * div_mask)[ds_idx].sum() / ((weights * div_mask)[ds_idx].sum() + eps)
            loss_ce = loss_ce_sp + loss_ce_ds

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_ce_nondiv = (raw_ce_loss * weights * non_div_mask).sum() / ((weights * non_div_mask).sum() + eps)
            loss_ce = loss_ce + loss_ce_nondiv
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, ignore_index=-1)

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(self, outputs, targets, indices, num_points, **kwargs):
        """
        SmoothL1 regression loss:
           - targets dicts must contain the key "points" containing a tensor of dim [nb_target_points, 2]
        """
        assert 'pred_points' in outputs
        # get indices
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5
            split_map = kwargs['div']
            div_thrs = self.div_thrs_dict[outputs['pq_stride']]
            div_mask = split_map > div_thrs
            loss_points_div = loss_points_raw * div_mask[idx].unsqueeze(-1)
            loss_points_div_sp = loss_points_div[pt_sp_idx].sum() / (len(pt_sp_idx) + eps)
            loss_points_div_ds = loss_points_div[pt_ds_idx].sum() / (len(pt_ds_idx) + eps)

            # loss on non-div regions
            non_div_mask = split_map <= div_thrs
            loss_points_nondiv = (loss_points_raw * non_div_mask[idx].unsqueeze(-1)).sum() / (non_div_mask[idx].sum() + eps)   

            # final point loss
            losses['loss_points'] = loss_points_div_sp + loss_points_div_ds + loss_points_nondiv
        else:
            losses['loss_points'] = loss_points_raw.sum() / num_points
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'points': self.loss_points,
        }
        assert loss in loss_map, f'{loss} loss is not defined'
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ Loss computation
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # compute the average number of target points accross all nodes, for normalization purposes
        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        return losses


class MLP(nn.Module):
    """
    Multi-layer perceptron (also called FFN)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_reduce=False, use_relu=True):
        super().__init__()
        # input_dim = hidden_dim = 256
        # output_dim = 2
        # num_layers = 3
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]  # [256, 128]
        else:
            h = [hidden_dim] * (num_layers - 1)  # [256, 256]
        # [input_dim] + h = [256, 256, 256]
        # h + [output_dim] = [256, 256, 2]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.use_relu = use_relu

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.use_relu:
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            else:
                x = layer(x)
        return x


def build_pet(args):
    device = torch.device(args.device)

    # build model
    num_classes = 1
    if args.backbone.startswith("vgg"):
        backbone = build_backbone_vgg(args)
    elif args.backbone.startswith("efficient"):
        backbone = build_backbone_efficient(args)
    elif args.backbone.startswith("resnet"):
        backbone = build_backbone_resnet(args)
    model = PET(
        backbone,
        num_classes=num_classes,
        args=args,
    )

    # build loss criterion
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
    losses = ['labels', 'points']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    return model, criterion
