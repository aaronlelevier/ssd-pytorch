import torch

from ssdmultibox.datasets import BATCH, NUM_CLASSES


CONF_THRESH = 0.1


class Predict:

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
        bbs, cats = cls.get_stacked_bbs_and_cats(preds)
        item_bbs, item_cats = bbs[index], cats[index]
        return cls.single_nms(cls_id, item_bbs, item_cats, conf_thresh)

    @staticmethod
    def get_stacked_bbs_and_cats(preds):
        """
        Returns all stacked bbs and cats from a single mini-batch of
        predictions

        Args:
            preds: mini-batch of model preds
        Returns:
            tuple (stacked bbs, stacked cats)
            example shapes:
                bbs - (4, 11640, 4)
                cats - (4, 11640, 20)
        """
        bbs = torch.cat([
            preds[i][j][0].reshape(BATCH, -1, 4)
            for j in range(6) for i in range(6)
        ], dim=1)

        cats = torch.cat([
            preds[i][j][1].reshape(BATCH, -1, NUM_CLASSES)[:,:,:-1]
            for j in range(6) for i in range(6)
        ], dim=1)

        # remove gradient tracking
        bbs.detach_()
        cats.detach_()

        return bbs, cats

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
        detected_cls_ids = torch.tensor([cls_id]).repeat(nms_count)
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
