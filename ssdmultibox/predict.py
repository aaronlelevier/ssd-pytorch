import torch

from ssdmultibox.datasets import BATCH, NUM_CLASSES, device, Bboxer

CONF_THRESH = 0.1


class Predict:

    @classmethod
    def predict_all(cls, preds, index=0, conf_thresh=CONF_THRESH):
        bbs = torch.empty(0, dtype=torch.float).to(device)
        scores = torch.empty(0, dtype=torch.float).to(device)
        cls_ids = torch.empty(0, dtype=torch.long).to(device)

        for c in range(NUM_CLASSES):
            single_preds = cls.single_predict(c, preds, index, conf_thresh)
            if single_preds:
                single_bbs, single_scores, single_cls_ids = single_preds
                bbs = torch.cat((bbs, single_bbs))
                scores = torch.cat((scores, single_scores))
                cls_ids = torch.cat((cls_ids, single_cls_ids))

        # sort descending what are the highest conf preds accross all classes
        sorted_scores, sorted_ids = scores.sort(0, descending=True)
        return bbs[sorted_ids], sorted_scores, cls_ids[sorted_ids]

    @classmethod
    def single_predict(cls, cls_id, preds, index=0, conf_thresh=CONF_THRESH):
        """
        Full predictions for a single class

        Args:
            cls_id (int): category_id
            preds: mini-batch preds from model
            index (int): index of batch item to choose
            conf_thresh (float):
                percent confidence threshold to filter detections by
        Returns:
            tuple(bbs, scores) or None
        """

        # this adds the AnchorBox offsets to the preds
        bbs_preds, cats_preds = preds
        stacked_anchor_boxes = torch.tensor(
            Bboxer.get_stacked_anchor_boxes(), dtype=bbs_preds.dtype).to(device)
        bbs_preds_w_offsets = stacked_anchor_boxes  + bbs_preds
        preds = (bbs_preds_w_offsets, cats_preds)

        bbs, cats = preds
        item_bbs, item_cats = bbs[index], cats[index]
        return cls.single_nms(cls_id, item_bbs, item_cats, conf_thresh)

    @classmethod
    def single_nms(cls, cls_id, item_bbs, item_cats, conf_thresh=CONF_THRESH):
        """
        Returns the NMS detections for a single image

        Args:
            cls_id (int): category id of object to detect
            item_bbs (2d array): [feature_maps, 4] bbs preds
            item_cats (2d array):[feature_maps, 20] one-hot cats preds
            conf_thresh (float):
                percent confidence threshold to filter detections by
        Returns:
            tuple ([nms_bbs, 4], [scores]) or None if no matches
        """
        item_bbs = item_bbs.detach()
        item_cats = item_cats.detach()

        cls_conf, cls_ids = item_cats.max(1)
        # per cls
        cls_conf_thresh_mask = cls_conf.gt(conf_thresh)
        cls_ids_gt_conf_thresh = cls_ids[cls_conf_thresh_mask]
        cls_conf_gt_conf_thresh = cls_conf[cls_conf_thresh_mask]
        bbs_gt_conf_thresh = item_bbs[cls_conf_thresh_mask]
        gt_conf_thresh_mask = cls_ids_gt_conf_thresh.eq(cls_id)

        boxes = bbs_gt_conf_thresh[gt_conf_thresh_mask]
        scores = cls_conf_gt_conf_thresh[gt_conf_thresh_mask]
        # exit if no matches
        if not scores.sum().item():
            return

        nms_ids, nms_count = cls.nms(boxes, scores)
        detected_ids = nms_ids[:nms_count]
        detected_cls_ids = torch.tensor([cls_id]).repeat(nms_count).to(device)
        return boxes[detected_ids], scores[detected_ids], detected_cls_ids

    # Original author: Francisco Massa:
    # https://github.com/fmassa/object-detection.torch
    # Ported to PyTorch by Max deGroot (02/01/2017)
    @staticmethod
    def nms(boxes, scores, overlap=0.5, top_k=200):
        """Apply non-maximum suppression at test time to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
            scores: (tensor) The class predscores for the img, Shape:[num_priors].
            overlap: (float) The overlap thresh for suppressing unnecessary boxes.
            top_k: (int) The Maximum number of box preds to consider.
        Return:
            The indices of the kept boxes with respect to num_priors.
        """

        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)  # sort in ascending order
        # I = I[v >= 0.01]
        idx = idx[-top_k:]  # indices of the top-k largest vals
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        # keep = torch.Tensor()
        count = 0
        while idx.numel() > 0:
            i = idx[-1]  # index of current largest val
            # keep.append(i)
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]  # remove kept element from view
            # load bboxes of next highest vals
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w*h
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter/union  # store result in iou
            # keep only elements with an IoU <= overlap
            idx = idx[IoU.le(overlap)]
        return keep, count
