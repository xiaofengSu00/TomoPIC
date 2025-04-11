import torch

def bbox_iou(box1, box2, GIOU=False, DIOU=False, eps=1e-6):
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 3] / 2, box1[:, 0] + box1[:, 3] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b1_z1, b1_z2 = box1[:, 2] - box1[:, 3] / 2, box1[:, 2] + box1[:, 3] / 2

    b2_x1, b2_x2 = box2[:, 0] - box2[:, 3] / 2, box2[:, 0] + box2[:, 3] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    b2_z1, b2_z2 = box2[:, 2] - box2[:, 3] / 2, box2[:, 2] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    inter = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1,
                                                                            min=0) * torch.clamp(
        inter_rect_z2 - inter_rect_z1, min=0)

    b1_vol = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_vol = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    union = b1_vol + b2_vol - inter + eps

    iou = inter / union
    if DIOU or GIOU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        cd = torch.max(b1_z2, b2_z2) - torch.min(b1_z1, b2_z1)
        if DIOU:
            c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + (
                        b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4
            return iou - rho2 / c2
        c_area = cw * ch * cd + eps
        return iou - (c_area - union) / c_area
    return iou


def bbox_eiou(box1, box2, GIOU=False, DIOU=False, EIOU=False, 
             Focal=False, alpha=1, gamma=0.5,eps=1e-6):
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 3] / 2, box1[:, 0] + box1[:, 3] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b1_z1, b1_z2 = box1[:, 2] - box1[:, 3] / 2, box1[:, 2] + box1[:, 3] / 2

    b2_x1, b2_x2 = box2[:, 0] - box2[:, 3] / 2, box2[:, 0] + box2[:, 3] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    b2_z1, b2_z2 = box2[:, 2] - box2[:, 3] / 2, box2[:, 2] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    inter = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1,
                                                                            min=0) * torch.clamp(
        inter_rect_z2 - inter_rect_z1, min=0)

    b1_vol = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_vol = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    union = b1_vol + b2_vol - inter + eps

    iou = inter / union
    if DIOU or GIOU or EIOU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        cd = torch.max(b1_z2, b2_z2) - torch.min(b1_z1, b2_z1)
        
        c2 = cw ** 2 + ch ** 2 + cd ** 2 + eps
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2 + (
                        b2_z1 + b2_z2 - b1_z1 - b1_z2) ** 2) / 4
        if DIOU:
            return iou - rho2 / c2
        
        elif EIOU:
            cw2 = torch.pow(cw ** 2 + eps, alpha)
            ch2 = torch.pow(ch ** 2 + eps, alpha)
            cd2 = torch.pow(cd ** 2 + eps, alpha)

            rho_r2 = (box1[:,3] - box2[:,3]) ** 2

            if Focal:
                return iou - (rho2/c2 + rho_r2/cw2 + rho_r2/ch2 + rho_r2/cd2),torch.pow(inter/union, gamma)
            else:
                return iou - (rho2/c2 + rho_r2/cw2 + rho_r2/ch2 + rho_r2/cd2)

        c_area = cw * ch * cd + eps
        return iou - (c_area - union) / c_area
    return iou




def cross_calculate_iou(box1, box2):
    """
        Calculate cross iou of two bounding box groups.
        Returns cross map with size [len(box1), len(box2)].
    """
    len1 = box1.size(0)
    len2 = box2.size(0)
    # repeat to same size and calculate
    BOX1 = box1.view(-1, 1, 4).repeat(1, len2, 1).view(-1, 4)
    BOX2 = box2.repeat(len1, 1)

    iou = bbox_iou(BOX1, BOX2)
    iou = iou.view(len1, len2).contiguous()

    return iou


def Non_Maximum_Suppression(boxes, nms_thres=0.3):
    """ Non maximum suppression
        Removes unnecessary check boxes
        Input:
            boxes: tensor of bounding boxes, size [num, 5], left out class for space
            nms_thres: threshold of iou

        Returns:
            left_index: index of chosen bounding boxes with one dimension
    """
    conf_slice = boxes[:, 4]
    _, index = torch.sort(conf_slice, descending=True)
    descending_boxes = boxes[index, :]

    iou = cross_calculate_iou(descending_boxes[:, :4], descending_boxes[:, :4])

    for i in range(descending_boxes.size(0)):
        if descending_boxes[i, 4] > 1e-6:
            remove_mask = iou[i, :] > nms_thres
            remove_mask[i] = False
            descending_boxes[remove_mask, 4] = 0

    left_index = index[descending_boxes[:, 4] > 1e-6]

    return left_index



def remove(boxes):
    """ Remove close candidates.
        If one candidate's location is in the detection area of another candidate with higher confidence, remove it.
        We choose 2 / 3 of prediction radius as the occupied area of one candidates.

        Input:
            boxes: tensor of candidates, size [num, 5], left out class for space

        Returns:
            left_index: index of chosen candidates with one dimension
    """
    conf_slice = boxes[:, 4]
    _, index = torch.sort(conf_slice, descending=True)
    descending_boxes = boxes[index, :]

    iou = cross_calculate_des(descending_boxes[:, :4], descending_boxes[:, :4])
    for i in range(descending_boxes.size(0)):
        if descending_boxes[i, 4] > 1e-6:
            remove_mask = iou[i, :] <= descending_boxes[i, 3] * 2 / 3
            remove_mask[i] = False
            descending_boxes[remove_mask, 4] = 0

    left_index = index[descending_boxes[:, 4] > 1e-6]
    # print('left_index: ', len(left_index))
    return left_index


def cross_calculate_des(box1, box2):
    """
        Calculate cross distance of two bounding box groups.
        Returns cross map with size [len(box1), len(box2)].
    """
    len1 = box1.size(0)
    len2 = box2.size(0)
    # repeat to same size and calculate
    BOX1 = box1.view(-1, 1, 4).repeat(1, len2, 1).view(-1, 4)
    BOX2 = box2.repeat(len1, 1)

    des = calculate_des(BOX1, BOX2)
    des = des.view(len1, len2).contiguous()

    return des


def calculate_des(box1, box2):
    """
        Calculate distance of two boxes tensor.
        Make sure all data > 0.
        Returns the distance of two bounding boxes.
    """
    return torch.sqrt(torch.pow(box2[:, 0] - box1[:, 0], 2) + torch.pow(box2[:, 1] - box1[:, 1], 2) + torch.pow(
        box2[:, 2] - box1[:, 2], 2))


