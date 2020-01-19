import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    
    def is_input_data_in_box3d(self, obj_list, d):
        x,y,z = d[0], d[1], d[2]
        for o in obj_list:
            '''
                xmin = o[4], ymin = o[5], xmax = o[6], ymax = [7]
            '''
            if o.cls_type == 'Car':
               o_corners = o.generate_corners3d()
               ymin = min(o_corners[0, 1], o_corners[4, 1])
               ymax = max(o_corners[0, 1], o_corners[4, 1])
               x1 = o_corners[0, 0]
               x2 = o_corners[1, 0]
               x3 = o_corners[2, 0]
               x4 = o_corners[3, 0]
               z1 = o_corners[0, 2]
               z2 = o_corners[1, 2]
               z3 = o_corners[2, 2]
               z4 = o_corners[3, 2]
               #print("y : %d" % y)
               #print("ymax : %d" % ymax)
               #print("ymin : %d" % ymin)
               if y <= ymax and y >= ymin:
                   if x <= max([x1, x2, x3, x4]) and x >= min([x1, x2, x3, x4]):
                       if z <= max([z1, z2, z3, z4]) and z >= min([z1, z2, z3, z4]):
                           return True               
        return False

    def fpnet_test(self, input_data, sample_id, rpn_scores_raw):
        input_data = input_data[0].cpu().numpy()
        sample_id = sample_id[0]
        import numpy as np
        import os
        fpnet_file = os.path.join('../data/KITTI/frustum_net/data/', '%06d.txt' % sample_id)
        import lib.utils.kitti_utils as kitti_utils
        obj_list = kitti_utils.get_objects_from_label(fpnet_file)
        for i in range(len(input_data)):
            d = input_data[i]
            if self.is_input_data_in_box3d(obj_list, d):
                rpn_scores_raw[0,i] = rpn_scores_raw[0,i] + 0.5
                #print("----------HAHAHAHAHAHAHAHA--------------> update rpn_scores_raw")
        return rpn_scores_raw
    
    def forward(self, input_data, sample_id=None):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)


                    # frustum update score
                    if cfg.FUSSION.FRUSTUM_AVAILABLE and (not sample_id == None):
                        rpn_scores_raw = self.fpnet_test(input_data['pts_input'], sample_id, rpn_scores_raw)
                    
                    
                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
