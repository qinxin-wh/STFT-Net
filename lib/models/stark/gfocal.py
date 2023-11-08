import torch
import torch.nn.functional as F


def quality_focal_loss(pred, score, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    # label, score = target

    # negatives are supervised by 0 quality score
    # pred_sigmoid = pred.sigmoid()
    # scale_factor = pred_sigmoid
    # zerolabel = scale_factor.new_zeros(pred.shape)
    # loss = F.binary_cross_entropy_with_logits(
    #     pred, zerolabel, reduction='none') * scale_factor.pow(beta)
    #
    # # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    # # bg_class_ind = pred.size(1)
    # # pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(1)
    # pos = torch.nonzero((label > 0), as_tuple=False).squeeze(1)  # 哪些样本是正样本
    # pos_label = label[pos].long()  # 正样本对应的具体类别
    # # positives are supervised by bbox quality (IoU) score
    # # a = pred_sigmoid[pos, pos_label]
    # a = pred_sigmoid[pos]
    pred_sigmoid = pred.sigmoid()
    scale_factor = score - pred_sigmoid
    s = scale_factor.abs().pow(beta)
    loss = F.binary_cross_entropy_with_logits(
        pred, score,
        reduction='none') * scale_factor.abs().pow(beta)

    loss = loss.mean()

    # loss = loss.sum(dim=1, keepdim=False)
    return loss

# a = torch.Tensor([0.2,0.7,0.9])
# b = torch.Tensor([0, 1, 1])
# c = torch.Tensor([0.33, 0.86, 0.66])
# out = quality_focal_loss(a, b, c)
