"""
PET model and criterion classes
"""
import torch
import torch.nn.functional as F
from torch import nn

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
    def __init__(self, backbone, num_classes, quadtree_layer='sparse', args=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.transformer = kwargs['transformer']
        hidden_dim = args.hidden_dim
        
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.coord_embed = MLP(hidden_dim, hidden_dim, 2, 3)

        self.pq_stride = [args.sparse_stride, args.dense_stride]
        self.feat_name = ['8x', '4x']
    
    def points_queris_embed(self, samples, strides=[8, 4], srcs=None, **kwargs):
        """
        Generate point query embedding during training
        """
        # dense position encoding at every pixel location

        # 원본 이미지의 position encoding 된 값. 이름을 왜 이따구로 해놨대
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]
        sum_query_embed=[]
        sum_point_queries=[]
        sum_query_feats=[]
        for idx, stride in enumerate(strides):
            if srcs[idx] is None:
                continue
            dec_win_w, dec_win_h = kwargs['dec_win_size']
            # get image shape
            input = samples.tensors
            image_shape = torch.tensor(input.shape[2:])
            shape = (image_shape + stride//2 -1) // stride

            # generate point queries
            shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
            shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2
            h, w = shift_x.shape

            # get point queries embedding
            query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
            bs, c = query_embed.shape[:2]
            query_embed = query_embed.view(bs, c, h, w)

            # query_embed = query_embed.flatten(2).permute(2,0,1)
            

            # get point queries features, equivalent to nearest interpolation
            shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
            query_feats = srcs[idx][:, :, shift_y_down, shift_x_down]
            query_feats = query_feats.view(bs, c, h, w)

            # window-rize
            # [win_vec, BxN, C]
            query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)
            # [BxN, win_vec, C]
            sum_query_embed.append(query_embed_win.permute(1,0,2))

            # points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
            # points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)
            # sum_point_queries.append(points_queries_win.permute(1,0,2))
            sum_point_queries.append(points_queries)

            query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
            sum_query_feats.append(query_feats_win.permute(1,0,2))

            # query_feats = query_feats.flatten(2).permute(2,0,1)
            # sum_query_feats.append(query_feats)
        query_embed = torch.vstack(sum_query_embed)
        query_feats = torch.vstack(sum_query_feats)
        points_queries = torch.vstack(sum_point_queries)
        return query_embed, points_queries, query_feats
    
    def points_queris_embed_inference(self, samples, strides=[8, 4], srcs=None, **kwargs):
        """
        Generate point query embedding during inference
        """
        # dense position encoding at every pixel location
        dense_input_embed = kwargs['dense_input_embed']
        bs, c = dense_input_embed.shape[:2]
        sum_query_embed=[]
        sum_point_queries=[]
        sum_query_feats=[]
        sum_vidx=[]
        for idx, stride in enumerate(strides):
            if srcs[idx] is None:
                continue
            dec_win_w, dec_win_h = kwargs['dec_win_size']
            # get image shape
            input = samples.tensors
            image_shape = torch.tensor(input.shape[2:])
            shape = (image_shape + stride//2 -1) // stride

            # generate points queries
            shift_x = ((torch.arange(0, shape[1]) + 0.5) * stride).long()
            shift_y = ((torch.arange(0, shape[0]) + 0.5) * stride).long()
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            points_queries = torch.vstack([shift_y.flatten(), shift_x.flatten()]).permute(1,0) # 2xN --> Nx2 x,y로 모양을 만드는 것이구나!
            h, w = shift_x.shape

            # get points queries embedding 
            query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]]
            bs, c = query_embed.shape[:2]

            # get points queries features, equivalent to nearest interpolation
            shift_y_down, shift_x_down = points_queries[:, 0] // stride, points_queries[:, 1] // stride
            query_feats = srcs[idx][:, :, shift_y_down, shift_x_down]

            # window-rize
            query_embed = query_embed.reshape(bs, c, h, w)
            query_embed_win = window_partition(query_embed, window_size_h=dec_win_h, window_size_w=dec_win_w)

            points_queries = points_queries.reshape(h, w, 2).permute(2, 0, 1).unsqueeze(0)
            points_queries_win = window_partition(points_queries, window_size_h=dec_win_h, window_size_w=dec_win_w)

            query_feats = query_feats.reshape(bs, c, h, w)
            query_feats_win = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)
            # BxN을 모두 앞으로
            sum_query_embed.append(query_embed_win.permute(1,0,2))
            sum_point_queries.append(points_queries_win.permute(1,0,2))
            sum_query_feats.append(query_feats_win.permute(1,0,2))

            # dynamic point query generation
            div = kwargs['div'][idx]
            div_win = window_partition(div.unsqueeze(1), window_size_h=dec_win_h, window_size_w=dec_win_w)
            valid_div = (div_win > 0.5).sum(dim=0)[:,0]
            v_idx = valid_div > 0
            sum_vidx.append(v_idx.unsqueeze(1))

        #Stack 후 BxN을 두번째로 (for v_indexing)
        query_embed = torch.vstack(sum_query_embed).permute(1,0,2)
        query_feats = torch.vstack(sum_query_feats).permute(1,0,2)
        points_queries = torch.vstack(sum_point_queries).permute(1,0,2)
        
        v_idx = torch.vstack(sum_vidx).squeeze(1)
        query_embed_win = query_embed[:, v_idx].permute(1,0,2)
        query_feats_win = query_feats[:, v_idx].permute(1,0,2)
        v_idx = v_idx.cpu()
        points_queries_win = points_queries[:, v_idx].reshape(-1, 2)
        return query_embed_win, points_queries_win, query_feats_win, v_idx
        
    def get_point_query(self, samples, features, **kwargs):
        """
        Generate point query
        """

        if kwargs['is_sparse']:
            src_s, _ = features[self.feat_name[0]].decompose()
        else:
            src_s = None
        if kwargs['is_dense']:
            src_d, _ = features[self.feat_name[1]].decompose()
        else:
            src_d = None

        # generate points queries and position embedding
        if 'train' in kwargs:
            query_embed, points_queries, query_feats = self.points_queris_embed(samples, self.pq_stride, [src_s, src_d], **kwargs)
            v_idx = None
        else:
            query_embed, points_queries, query_feats, v_idx = self.points_queris_embed_inference(samples, self.pq_stride, [src_s, src_d], **kwargs)

        out = (query_embed, points_queries, query_feats, v_idx)
        return out
    
    def predict(self, samples, points_queries, hs, **kwargs):
        """
        Crowd prediction
        """
        outputs_class = self.class_embed(hs)
        # normalize to 0~1
        outputs_offsets = (self.coord_embed(hs).sigmoid() - 0.5) * 2.0

        # normalize point-query coordinates
        img_shape = samples.tensors.shape[-2:]
        img_h, img_w = img_shape

        points_queries = points_queries.float().cuda()
        points_queries[:, 0] /= img_h
        points_queries[:, 1] /= img_w

        # rescale offset range during testing
        if 'test' in kwargs:
            outputs_offsets[...,0] /= (img_h / 256)
            outputs_offsets[...,1] /= (img_w / 256)
            
        outputs_points = outputs_offsets + points_queries
        out = {'pred_logits': outputs_class, 'pred_points': outputs_points, 'img_shape': img_shape, 'pred_offsets': outputs_offsets}
    
        out['points_queries'] = points_queries
        out['pq_stride'] = self.pq_stride
        return out

    def forward(self, samples, features, **kwargs):
        # encode_src, src_pos_embed, mask = context_info

        # get points queries for transformer
        # pqs=([1024, 8, 256], [1024, 2], [8, 256, 32, 32], None)
        pqs = self.get_point_query(samples, features, **kwargs)

        # point querying
        kwargs['pq_stride'] = self.pq_stride
        
        # encode_src: 인코딩된 src(from feature), pqs: splitter에 의해 잘린 sample(image)와 feature
        # encode_src: [8, 256, 32, 32], src_pos_embed: [8, 256, 32, 32], mask: [8, 32, 32], pqs, img_shape: [256, 256]
        hs = self.transformer(pqs, **kwargs)
        # hs = self.transformer(encode_src, src_pos_embed, mask, pqs, img_shape=samples.tensors.shape[-2:], **kwargs)
        # hs = torch.Size([2, 8, 1024, 256]) 가 나와야 한다.
        # prediction
        points_queries = pqs[1]
        outputs = self.predict(samples, points_queries, hs, **kwargs)
        return outputs
    
# backbone, num_classes=1, args=args
class PET(nn.Module):
    """ 
    Point quEry Transformer
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
        enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]  # encoder window size
        args.enc_layers = len(enc_win_list)

        # quadtree splitter
        context_patch = (128, 64)
        context_w, context_h = context_patch[0]//int(self.encode_feats[:-1]), context_patch[1]//int(self.encode_feats[:-1])  # context_w = 128//8 = 16, context_h = 64//8 = 8
        self.quadtree_splitter = nn.Sequential(
            nn.AvgPool2d((context_h, context_w), stride=(context_h, context_w)),  # kernel_size = (8, 16), stride = (8, 16)
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

        # point-query quadtree
        args.sparse_stride, args.dense_stride = 8, 4  # point-query stride
        transformer = build_encoder(args)

        # 원래 기본 이미지가 BasePETCount에 사용된다! 즉 Decoder에 사용된다.
        self.encoder = BasePETCount(backbone, num_classes, quadtree_layer='sparse', args=args, transformer=transformer)

    def compute_loss(self, outputs, criterion, targets, epoch, samples):
        """
        Compute loss, including:
            - point query loss (Eq. (3) in the paper)
            - quadtree splitter loss (Eq. (4) in the paper)
        """
        output = outputs['both']
        weight_dict = criterion.weight_dict
        warmup_ep = 5
        # compute loss
        if epoch >= warmup_ep:
            # output.keys(): 
            loss_dict_both = criterion(output, targets, div=[outputs['split_map_sparse'], outputs['split_map_dense']],test=True)
        else:
            loss_dict_both = criterion(output, targets)

        # both point queries loss
            
        loss_dict_both = {k+'_bt':v for k, v in loss_dict_both.items()}
        weight_dict_both = {k+'_bt':v for k,v in weight_dict.items()}
        # weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_points': args.point_loss_coef}
        loss_pq = sum(loss_dict_both[k] * weight_dict_both[k] for k in loss_dict_both.keys() if k in weight_dict_both)

        # point queries loss
        losses = loss_pq

        # update loss dict and weight dict
        loss_dict = dict()
        loss_dict.update(loss_dict_both)

        weight_dict = dict()
        weight_dict.update(weight_dict_both)

        # quadtree splitter loss
        den = torch.tensor([target['density'] for target in targets])   # crowd density
        bs = len(den)
        ds_idx = den < 2 * self.encoder.pq_stride[0]  # dense regions index
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
        # samples 입력: NestedTensor(tensors, mask) --> tensors: (B, 3, 256, 256), mask: (B, 256, 256)
        # output: out = {'4x': NestedTensor(features_fpn_4x, mask_4x), --> features_fpn_4x.shape: (B, 256, 64, 64), mask_4x.shape: (B, 64, 64)
        #                '8x': NestedTensor(features_fpn_8x, mask_8x)} --> features_fpn_8x.shape: (B, 256, 32, 32), mask_8x.shape: (B, 32, 32)
        # 이때, 각 feature map들은 position_embedding이 완료된 상태로 튀어나온다!

        # positional embedding
        # 원래 기본 이미지에 대해서 position_embedding을 진행한다(like ViT)
        # samples: {tensors: [8, 3, 256, 256], mask: [8, 256, 256]}
        dense_input_embed = self.pos_embed(samples)
        # [8, 256, 256, 256]
        kwargs['dense_input_embed'] = dense_input_embed

        # feature projection
        # FPN에서 hidden_size를 256으로 만들었기 때문에, 256 channel로 나오게 된다.
        # 256 channel로 나온 feature map을 hidden_dim으로 projection 시킨다.
        # base의 hidden_dim은 256이므로, feature를 새로운 256 size로 재구성한다고 보아도 된다...? 이 과정이 필요할까?
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
        src, mask = features[self.encode_feats].decompose()

        assert mask is not None

        # apply quadtree splitter
        bs, _, src_h, src_w = src.shape  # bs = 8, src_h = 32, src_w = 32
        kwargs['batch_size'] = bs
        sp_h, sp_w = src_h, src_w  # sp_h = 32, sp_w = 32
        ds_h, ds_w = int(src_h * 2), int(src_w * 2)  # ds_h = 64, ds_w = 64
        split_map = self.quadtree_splitter(src)  # [8, 1, 2, 4]
        split_map_dense = F.interpolate(split_map, (ds_h, ds_w)).reshape(bs, -1)  # interpolate(split_map) = [8, 1, 64, 64] / interpolate(split_map).reshape = [8, 4096]
        split_map_sparse = 1 - F.interpolate(split_map, (sp_h, sp_w)).reshape(bs, -1)  # interpolate(split_map) = [8, 1, 32, 32] / interpolate(split_map).reshape = [8, 1024]
        
        
        kwargs['div'] = [split_map_sparse.reshape(bs, sp_h, sp_w), split_map_dense.reshape(bs, ds_h, ds_w)]

        kwargs['is_sparse'] = (split_map_sparse > 0.5).sum() > 0

        kwargs['is_dense'] = (split_map_dense > 0.5).sum() > 0

        kwargs['dec_win_size'] = [16, 8]

        outputs_both = self.encoder(samples, features, **kwargs)

        # format outputs
        outputs = dict()
        outputs['both'] = outputs_both
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
        out_both = outputs['both']
        thrs = 0.5  # inference threshold
        # process both point queries
        out_scores = torch.nn.functional.softmax(out_both['pred_logits'], -1)[..., 1]
        valid_both = out_scores > thrs
        index_both = valid_both.cpu()

        # format output
        div_out = dict()
        output_names = out_both.keys()

        for name in list(output_names):
            if 'pred' in name:
                div_out[name] = out_both[name][index_both].unsqueeze(0)
            else:
                div_out[name] = out_both[name]
        
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
            target_idx = -1
            for tmp_idx in range(2):
                if raw_ce_loss.shape[-1] == kwargs['div'][tmp_idx].shape[-1]:
                    target_idx = tmp_idx
                    break

            if target_idx == -1:
                split_map = torch.vstack([kwargs['div'][0].permute(1,0), kwargs['div'][1].permute(1,0)]).permute(1,0)
            else:
                split_map = kwargs['div'][target_idx]
            div_thrs = self.div_thrs_dict[outputs['pq_stride'][target_idx]]
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
        src_logits = outputs['pred_logits']
        target_points = torch.cat([t['points'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # compute regression loss
        losses = {}
        img_shape = outputs['img_shape']
        img_h, img_w = img_shape
        target_points[:, 0] /= img_h
        target_points[:, 1] /= img_w
        loss_points_raw = F.smooth_l1_loss(src_points, target_points, reduction='none')

        if 'div' in kwargs:
            losses['loss_points'] = 0
            # split_map = torch.vstack([kwargs['div'][0].permute(1,0), kwargs['div'][1].permute(1,0)]).permute(0,1)
            # get sparse / dense index
            den = torch.tensor([target['density'] for target in targets])
            den_sort = torch.sort(den)[1]
            img_ds_idx = den_sort[:len(den_sort)//2]
            img_sp_idx = den_sort[len(den_sort)//2:]
            pt_ds_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_ds_idx])
            pt_sp_idx = torch.cat([torch.where(idx[0] == bs_id)[0] for bs_id in img_sp_idx])

            # dual supervision for sparse/dense images
            eps = 1e-5

            target_idx = -1
            for tmp_idx in range(2):
                if src_logits.shape[1] == kwargs['div'][tmp_idx].shape[-1]:
                    target_idx = tmp_idx
                    break

            if target_idx == -1:
                split_map = torch.vstack([kwargs['div'][0].permute(1,0), kwargs['div'][1].permute(1,0)]).permute(1,0)
            else:
                split_map = kwargs['div'][target_idx]

            div_thrs = self.div_thrs_dict[outputs['pq_stride'][target_idx]]
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
        indices = self.matcher(outputs, targets, **kwargs)

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
        self.num_layers = num_layers
        if is_reduce:
            h = [hidden_dim//2**i for i in range(num_layers - 1)]
        else:
            h = [hidden_dim] * (num_layers - 1)
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
    # 여기서 넘어가는 backbone은 그냥 backbone만 있는것이 아닌, Joiner, 즉 position embedding이 포함된 모델이다!
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
