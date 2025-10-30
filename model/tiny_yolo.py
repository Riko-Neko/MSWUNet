# tiny_yolo.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- utils ----
def bbox_iou(box1, box2, eps=1e-6):
    """
    box: (x_center, y_center, w, h) in absolute or normalized coords (same system).
    returns IoU scalar or vector.
    """
    # convert to (x1,y1,x2,y2)
    b1_x1 = box1[...,0] - box1[...,2] / 2
    b1_y1 = box1[...,1] - box1[...,3] / 2
    b1_x2 = box1[...,0] + box1[...,2] / 2
    b1_y2 = box1[...,1] + box1[...,3] / 2

    b2_x1 = box2[...,0] - box2[...,2] / 2
    b2_y1 = box2[...,1] - box2[...,3] / 2
    b2_x2 = box2[...,0] + box2[...,2] / 2
    b2_y2 = box2[...,1] + box2[...,3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1).clamp(min=0) * (b1_y2 - b1_y1).clamp(min=0)
    area2 = (b2_x2 - b2_x1).clamp(min=0) * (b2_y2 - b2_y1).clamp(min=0)
    union = area1 + area2 - inter_area + eps
    return inter_area / union

# ---- Tiny YOLO model ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class TinyYOLO(nn.Module):
    """
    Very small YOLO-style single-scale detector.
    Output: (B, S, S, B*5 + C) as a flattened tensor: (B, output_dim, S, S)
    """
    def __init__(self, num_classes=2, S=7, B=2, base_channels=32):
        super().__init__()
        self.S = S
        self.B = B
        self.C = num_classes
        # backbone: extremely small
        c = base_channels
        self.backbone = nn.Sequential(
            ConvBlock(1, c, k=3, s=1, p=1),    # c x H x W
            nn.MaxPool2d(2,2),                 # c x H/2 x W/2
            ConvBlock(c, c*2, k=3, s=1, p=1),  # 2c
            nn.MaxPool2d(2,2),                 # 2c x H/4
            ConvBlock(c*2, c*4, k=3, s=1, p=1),# 4c
            nn.MaxPool2d(2,2),                 # 4c x H/8
            ConvBlock(c*4, c*8, k=3, s=1, p=1),# 8c
            nn.MaxPool2d(2,2),                 # 8c x H/16
            ConvBlock(c*8, c*8, k=3, s=1, p=1),# 8c
            # optionally more convs
        )
        # proj to grid S x S: we'll use adaptive pool to force SxS
        self.adaptive = nn.AdaptiveAvgPool2d((S, S))
        out_dim = B*5 + self.C
        # head: 1x1 conv to map channels -> output_dim
        self.head = nn.Conv2d(c*8, out_dim, kernel_size=1, stride=1, padding=0)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,3,H,W)
        Bn = x.shape[0]
        f = self.backbone(x)            # (B, c*8, H', W')
        f = self.adaptive(f)            # (B, c*8, S, S)
        out = self.head(f)              # (B, out_dim, S, S)
        # reshape to (B, S, S, out_dim)
        out = out.permute(0,2,3,1).contiguous()
        return out

# ---- Loss (YOLOv1 style simplified) ----
class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # classification uses cross entropy per cell (could be MSE as original YOLO)
        self.cls_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, preds, targets):
        """
        preds: (B, S, S, B*5 + C)
        targets: list of length B; each is a tensor of shape (num_gt, 5) in normalized coords (cx,cy,w,h,class_idx)
                 where coords are in [0,1] relative to image. You can also encode as tensor target_map.
        For simplicity, we convert targets into a target map tensor of shape (B, S, S, 5+B + C) or use mapping here.
        We'll assume targets_map: (B, S, S, 5) where 5 = [obj_flag, cx_cell, cy_cell, w, h] + class separately.
        To keep interface simple, require targets_map: tensor shape (B, S, S, 5 + C)
        Format (per cell):
          - obj = 0/1
          - bbox: cx_cell_rel (in [0,1] relative to cell), cy..., w(absolute relative to image), h(...)
          - class one-hot length C
        """
        # Expect user to pre-build targets_map for batching convenience.
        target_map = targets  # (B, S, S, 5 + C)
        device = preds.device
        Bn = preds.shape[0]
        S = self.S
        Bboxes = self.B
        C = self.C

        # split preds
        # preds[..., :B*5] contains boxes and confidences
        pred_box_conf = preds[..., :Bboxes*5].contiguous().view(Bn, S, S, Bboxes, 5)
        pred_cls_logits = preds[..., Bboxes*5:]  # (B, S, S, C)

        # target parsing
        obj_mask = target_map[..., 0].unsqueeze(-1)  # (B,S,S,1)
        true_cls = target_map[..., 5:]  # (B,S,S,C) one-hot
        true_box = target_map[..., 1:5]  # (B,S,S,4) (cx_in_cell, cy_in_cell, w, h)

        # For each cell, decide responsible bbox: choose the one with higher IoU between pred_box and gt
        # convert pred boxes relative to cell to absolute normalized coords:
        # pred_box[...,0:2] are offsets in cell coordinate (we assume network predicted them directly),
        # but we'll treat predictions as already in [0,1] relative to cell: user must ensure target encoding matches.
        # For simplicity we assume both pred and target share same encoding system.

        # compute IoU between each pred bbox and true_box (broadcast)
        # pred_box_conf[..., :4] = (cx,cy,w,h)
        pred_boxes = pred_box_conf[..., :4]  # (B,S,S,B,4)
        # expand true_box to match B dimension
        true_box_exp = true_box.unsqueeze(3).expand_as(pred_boxes)
        ious = bbox_iou(pred_boxes, true_box_exp)  # (B,S,S,B)

        # responsible bbox: argmax ious along B
        best_bbox_idx = torch.argmax(ious, dim=-1)  # (B,S,S)
        best_bbox_onehot = F.one_hot(best_bbox_idx, num_classes=Bboxes).to(device)  # (B,S,S,B)

        # Create masks
        obj_mask_bool = (obj_mask.squeeze(-1) > 0.5)  # (B,S,S)
        obj_mask_f = obj_mask_bool.float().unsqueeze(-1)  # (B,S,S,1)

        # --- Localization loss (only for responsible boxes where object exists) ---
        # select responsible pred boxes
        # pred responsible boxes:
        resp_pred_boxes = (pred_boxes * best_bbox_onehot.unsqueeze(-1)).sum(dim=3)  # (B,S,S,4)
        # sqrt for w,h as original YOLO
        pred_xy = resp_pred_boxes[..., :2]
        pred_wh = resp_pred_boxes[..., 2:].clamp(min=1e-6)
        pred_sqrt_wh = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh))

        true_xy = true_box[..., :2]
        true_wh = true_box[..., 2:].clamp(min=1e-6)
        true_sqrt_wh = torch.sign(true_wh) * torch.sqrt(torch.abs(true_wh))

        loc_loss = F.mse_loss(pred_xy * obj_mask_f, true_xy * obj_mask_f, reduction='sum')
        loc_loss += F.mse_loss(pred_sqrt_wh * obj_mask_f, true_sqrt_wh * obj_mask_f, reduction='sum')

        # --- Confidence loss ---
        pred_conf = pred_box_conf[..., 4]  # (B,S,S,B)
        # object confidence target: IoU for responsible bbox, 0 for others
        # build conf_target
        ious_for_resp = ious * best_bbox_onehot  # zeros except resp bbox
        conf_target = ious_for_resp.sum(dim=-1)  # (B,S,S) IoU value for responsible bbox
        # expand conf_target for per-bbox comparison
        conf_target_exp = torch.zeros_like(pred_conf)
        # set responsible bbox target to conf_target where obj exists
        conf_target_exp = conf_target_exp.to(device)
        conf_target_exp[obj_mask_bool] = 0.0  # keep zero default
        # fill responsible positions
        # create mask for responsible bbox positions
        resp_mask = best_bbox_onehot.bool()  # (B,S,S,B)
        # assign conf_target into resp positions
        val = conf_target.unsqueeze(-1).expand_as(pred_conf)
        conf_target_exp = torch.where(resp_mask, val, torch.zeros_like(val))

        # object (positive) loss
        obj_loss = F.mse_loss(pred_conf * obj_mask_f.expand_as(pred_conf), conf_target_exp * obj_mask_f.expand_as(pred_conf), reduction='sum')
        # no-object loss: for boxes not responsible and cells without object
        noobj_mask = ~resp_mask  # where not responsible
        noobj_loss = F.mse_loss(pred_conf[noobj_mask], torch.zeros_like(pred_conf[noobj_mask]), reduction='sum')

        # apply lambda weights
        conf_loss = obj_loss + self.lambda_noobj * noobj_loss

        # --- Classification loss (only for cells with object) ---
        # preds: logits shape (B,S,S,C). We'll compute CE only for object cells.
        # gather class labels from true_cls one-hot -> integer label
        true_cls_idx = true_cls.argmax(dim=-1)  # (B,S,S)
        # flatten for CE
        pred_cls_flat = pred_cls_logits[obj_mask_bool]  # (num_obj_cells, C)
        true_cls_flat = true_cls_idx[obj_mask_bool]    # (num_obj_cells,)
        if pred_cls_flat.numel() == 0:
            cls_loss = torch.tensor(0., device=device)
        else:
            cls_loss = self.cls_loss_fn(pred_cls_flat, true_cls_flat)

        total_loss = self.lambda_coord * loc_loss + conf_loss + cls_loss
        loss_dict = {
            'total': total_loss,
            'loc': loc_loss.detach(),
            'conf': conf_loss.detach(),
            'cls': cls_loss.detach()
        }
        return total_loss, loss_dict

# ---- Inference helpers ----
def decode_predictions(preds, S, B, conf_thresh=0.25):
    """
    preds: (B, S, S, B*5 + C) network output raw (no activation assumed)
    We'll sigmoid the confs and bbox xy offsets if necessary. For simplicity, assume network outputs:
      - bbox cx,cy are relative offsets in cell after sigmoid
      - w,h are in [0,1] predicted directly (maybe via sigmoid too)
    This function returns list per image of detections [x1,y1,x2,y2,score,class]
    Coordinates normalized in [0,1]
    """
    batch = preds.shape[0]
    device = preds.device
    out = []
    for i in range(batch):
        p = preds[i]  # S,S,*
        boxes = []
        # parse
        box_conf = p[..., :B*5].view(S, S, B, 5)  # cx,cy,w,h,conf
        cls_logits = p[..., B*5:]  # S,S,C
        cls_prob = F.softmax(cls_logits, dim=-1)  # S,S,C
        # iterate cells
        for ys in range(S):
            for xs in range(S):
                for b in range(B):
                    cb = box_conf[ys,xs,b]
                    conf = torch.sigmoid(cb[4])  # objectness
                    if conf.item() < conf_thresh:
                        continue
                    # bbox center in normalized image coordinates:
                    # cell top-left = (xs / S, ys / S)
                    cx = (xs + torch.sigmoid(cb[0])) / S
                    cy = (ys + torch.sigmoid(cb[1])) / S
                    w = cb[2].clamp(1e-6)
                    h = cb[3].clamp(1e-6)
                    # final class score: conf * class_prob
                    class_probs = cls_prob[ys,xs]
                    class_score, class_idx = torch.max(class_probs, dim=-1)
                    score = conf * class_score
                    if score.item() < conf_thresh:
                        continue
                    x1 = (cx - w/2).item()
                    y1 = (cy - h/2).item()
                    x2 = (cx + w/2).item()
                    y2 = (cy + h/2).item()
                    boxes.append([x1,y1,x2,y2, score.item(), int(class_idx.item())])
        out.append(boxes)
    return out

# Simple NMS (uses torchvision if available)
def non_max_suppression(detections, iou_threshold=0.5):
    """
    detections: list of [x1,y1,x2,y2,score,class]
    returns filtered list
    """
    try:
        from torchvision.ops import nms
        kept = []
        if len(detections) == 0:
            return []
        det = torch.tensor(detections)
        boxes = det[:, :4]
        scores = det[:, 4]
        # handle per-class NMS
        classes = det[:,5].long()
        final = []
        for c in classes.unique():
            mask = (classes == c)
            boxes_c = boxes[mask]
            scores_c = scores[mask]
            idxs = nms(boxes_c, scores_c, iou_threshold)
            sel = torch.arange(len(detections))[mask.nonzero(as_tuple=False).squeeze(1)][idxs]
            for s in sel:
                final.append(detections[int(s)])
        return final
    except Exception:
        # fallback: naive NMS per class (O(n^2))
        out=[]
        byclass = {}
        for d in detections:
            byclass.setdefault(d[5], []).append(d)
        for cls, dets in byclass.items():
            dets = sorted(dets, key=lambda x: x[4], reverse=True)
            keep=[]
            while dets:
                top = dets.pop(0)
                keep.append(top)
                rem=[]
                for d in dets:
                    # compute IoU
                    xa1,ya1,xa2,ya2 = top[0],top[1],top[2],top[3]
                    xb1,yb1,xb2,yb2 = d[0],d[1],d[2],d[3]
                    inter_x1 = max(xa1, xb1)
                    inter_y1 = max(ya1, yb1)
                    inter_x2 = min(xa2, xb2)
                    inter_y2 = min(ya2, yb2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter = inter_w * inter_h
                    area_a = max(0, xa2-xa1)*max(0,ya2-ya1)
                    area_b = max(0, xb2-xb1)*max(0, yb2-yb1)
                    iou = inter / (area_a + area_b - inter + 1e-6)
                    if iou <= iou_threshold:
                        rem.append(d)
                dets = rem
            out.extend(keep)
        return out

# ---- Quick usage example ----
if __name__ == "__main__":
    # small test
    model = TinyYOLO(num_classes=3, S=7, B=2, base_channels=16)  # base_channels tune reduces params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # fake input
    x = torch.randn(2,3,224,224).to(device)
    preds = model(x)  # (B,S,S,out_dim)
    print("preds shape:", preds.shape)
    # build a fake target_map: (B,S,S,5+C)
    Bbatch = preds.shape[0]
    S=7
    C=3
    target_map = torch.zeros(Bbatch, S, S, 5 + C).to(device)
    # create one box in image 0 at center cell
    target_map[0, 3, 3, 0] = 1.0  # obj
    target_map[0, 3, 3, 1:5] = torch.tensor([0.5,0.5,0.2,0.3]).to(device)  # cx_rel_in_cell, cy..., w,h
    target_map[0, 3, 3, 5+1] = 1.0  # class 1
    loss_fn = YOLOv1Loss(S=S, B=2, C=C)
    loss, parts = loss_fn(preds, target_map)
    print("loss:", loss.item(), parts)