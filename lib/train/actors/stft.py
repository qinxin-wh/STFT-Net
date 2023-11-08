from . import STARKSActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
import torch.nn.functional as F
from lib.models.stark.gfocal import quality_focal_loss


class STFTActor(STARKSActor):
    """ Actor for training the STFT(Stage2)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective, loss_weight, settings)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=False, run_cls_head=True)

        # process the groundtruth label
        labels = data['label'].view(-1)  # (batch, ) 0 or 1
        gt_bboxes = data['search_anno']

        loss, status = self.compute_losses(out_dict, labels, gt_bboxes[0])

        return loss, status

    def compute_losses(self, pred_dict, labels, gt_bbox, return_status=True):
        pred_boxes = pred_dict['pred_boxes']
        num_queries = pred_boxes.size(0)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        loss_giou, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        loss_reg = self.loss_weight["reg"] * loss_giou
        loss_cls = self.loss_weight["cls"] * quality_focal_loss(pred_dict["pred_logits"].view(-1), iou)
        loss = loss_cls + loss_reg
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {
                "cls_loss": loss_cls.item(),
                "reg_loss": loss_reg.item(),
                "iou": mean_iou.item()
            }
            return loss, status
        else:
            return loss
