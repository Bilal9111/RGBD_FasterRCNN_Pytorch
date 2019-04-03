import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic): # This may not need to change (except the inits)
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
























    def forward(self, im_data_a, im_info_a, gt_boxes_a, num_boxes_a, im_data_b, im_info_b, gt_boxes_b, num_boxes_b):

        #print(torch.equal(im_info_a,im_info_b))
        #print(torch.equal(gt_boxes_a,gt_boxes_b))
        #print(torch.equal(num_boxes_a,num_boxes_b))
        #print(im_data_a.size(0))
        #print(im_data_b.size(0))
        
        # Main here
        batch_size = im_data_a.size(0)
        im_info = im_info_a.data
        gt_boxes = gt_boxes_a.data
        num_boxes = num_boxes_a.data
        
      

        """ feed image data to base model to obtain base feature map """
        
        base_feat_a = self.RCNN_base_a(im_data_a) # feeding the data in the RCNN_base to get the feature maps
        """        
        x = self.RCNN_base_a_mod[0](im_data_a)
        x = self.RCNN_base_a_mod[1](x)
        x = self.RCNN_base_a_mod[2](x)
        x = self.RCNN_base_a_mod[3](x)
        x = self.RCNN_base_a_mod[4](x)
        x = self.RCNN_base_a_mod[5](x) # layer 2
        x = torch.cat((x,x),1)
        base_feat_a = self.RCNN_base_a_mod[6](x)      
        """
        
        base_feat_b = self.RCNN_base_b(im_data_b) # feeding the data in the RCNN_base to get the feature maps
             
        base_feat = base_feat_a # torch.cat((base_feat_a,base_feat_b),1)         
        
        """ feed base feature map tp RPN to obtain rois """
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes) # feeding the the feature maps created and the rest of image info to rpn to get the PRREDICTED rois

       

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes) # refining the rois if in training
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
             
            rois_label = Variable(rois_label.view(-1).long()) # The predicted roi labels
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        
        
        
        
        
        
        
        
        
        # do roi pooling based on predicted rois
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy_a = _affine_grid_gen(rois.view(-1, 5), base_feat_a.size()[2:], self.grid_size)
            grid_yx_a = torch.stack([grid_xy_a.data[:,:,:,1], grid_xy_a.data[:,:,:,0]], 3).contiguous()
            pooled_feat_a = self.RCNN_roi_crop(base_feat_a, Variable(grid_yx_a).detach())
            
            grid_xy_b = _affine_grid_gen(rois.view(-1, 5), base_feat_b.size()[2:], self.grid_size)
            grid_yx_b = torch.stack([grid_xy_b.data[:,:,:,1], grid_xy_b.data[:,:,:,0]], 3).contiguous()
            pooled_feat_b = self.RCNN_roi_crop(base_feat_b, Variable(grid_yx_b).detach())
            
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat_a = F.max_pool2d(pooled_feat_a, 2, 2)
                pooled_feat_b = F.max_pool2d(pooled_feat_b, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat_a = self.RCNN_roi_align(base_feat_a, rois.view(-1, 5))
            pooled_feat_b = self.RCNN_roi_align(base_feat_b, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_a = self.RCNN_roi_pool(base_feat_a, rois.view(-1,5))
            pooled_feat_b = self.RCNN_roi_pool(base_feat_b, rois.view(-1,5))

        
        
        
        # feed pooled features to top model
        #print("testing starts")
        #print(pooled_feat_a.shape)
        #print(pooled_feat_b.shape)
        pooled_feat = self._head_to_tail(pooled_feat_a,pooled_feat_b)
        #print(pooled_feat.shape)
        
        pooled_feat = self.fc_downsample_1(pooled_feat)
        pooled_feat = F.relu(pooled_feat)
        #pooled_feat = self.fc_downsample_2(pooled_feat)
        #pooled_feat = F.relu(pooled_feat)
        #pooled_feat = self.fc_downsample_3(pooled_feat)
        #pooled_feat = F.relu(pooled_feat)
        #pooled_feat = self.fc_downsample_4(pooled_feat)
        #pooled_feat = F.relu(pooled_feat)
        #pooled_feat = self.fc_downsample_5(pooled_feat)
        #pooled_feat = F.relu(pooled_feat)
        pooled_feat = self.fc_downsample_6(pooled_feat)
        pooled_feat = F.relu(pooled_feat)
        
        


        
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)        
        #print(bbox_pred.shape)
        
        
        if self.training and not self.class_agnostic:            
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            rois_label_view = rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)            
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label_view)
            bbox_pred = bbox_pred_select.squeeze(1)
        
        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        
        

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label





















    def _init_weights(self): # This may not need to change
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)        
        normal_init(self.fc_downsample_1, 0, 0.001, cfg.TRAIN.TRUNCATED)       
        #normal_init(self.fc_downsample_2, 0, 0.001, cfg.TRAIN.TRUNCATED)       
        #normal_init(self.fc_downsample_3, 0, 0.001, cfg.TRAIN.TRUNCATED)       
        #normal_init(self.fc_downsample_4, 0, 0.001, cfg.TRAIN.TRUNCATED)       
        #normal_init(self.fc_downsample_5, 0, 0.001, cfg.TRAIN.TRUNCATED)       
        normal_init(self.fc_downsample_6, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self): # This may not need to change too
        self._init_modules() # predefined function in nn.Module file
        self._init_weights() 
