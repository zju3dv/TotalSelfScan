import numpy as np


class Evaluator:
    def __init__(self):
        self.mask_intersection = []
        self.mask_iou = []

    def compute_iou(self, occ1, occ2):
        ''' Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        '''
        occ1 = np.asarray(occ1)
        occ2 = np.asarray(occ2)

        # Put all data in second dimension
        # Also works for 1-dimensional data
        if occ1.ndim >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if occ2.ndim >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Compute IOU
        area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
        area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)
        iou = (area_intersect / area_union)

        return iou

    def evaluate(self, output, batch):
        mask_gt = output['mask_gt'].detach().cpu().numpy()
        mask_pred = output['mask_pred'].detach().cpu().numpy()
        mask_intersection = (mask_gt == mask_pred).mean()
        mask_iou = self.compute_iou(mask_gt, mask_pred)
        self.mask_intersection.append(mask_intersection)
        self.mask_iou.append(mask_iou)

    def summarize(self):
        mask_intersection = np.mean(self.mask_intersection)
        mask_iou = np.mean(self.mask_iou)
        self.mask_intersection = []
        self.mask_iou = []
        result = {'mask_intersection': mask_intersection, 'mask_iou': mask_iou}
        print(result)
        return result
