from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation import SemSegEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import numpy as np
import os
from collections import OrderedDict
from detectron2.utils.file_io import PathManager
import json
import torch
import itertools

class CustomSemSegEvaluator(SemSegEvaluator):
    def __init__(self, model, dataset_name, distributed, output_dir=None, custom_metric=None):
        super().__init__(dataset_name, distributed, output_dir)
        self.ov_classifier = model.ov_classifier # san linear classifier to project the pixel embedding into the vocabulary space

    def label_semantic_metric(self, conf_matrix, b_conf_matrix):
        """
        Custom metric that computes the mean semantic pixel similarity between the predicted class and the ground truth class.
        Semantic similarity between two classes is defined as dot product of the encoded class vectors.

        Args:
            conf_matrix (np.ndarray): confusion matrix of shape (num_classes, num_classes)

        Returns:
            float: mean semantic pixel similarity
        """
        if not self.ov_classifier.cache:
            raise ValueError("Classifier cache is empty. Run inference on the dataset to populate the cache with the encoded labels.")
        
        num_classes = conf_matrix.shape[0]
        assert num_classes - 1 == self.ov_classifier.cache[self._dataset_name].shape[0], "Number of classes in the dataset does not match the number of classes in the classifier cache (vocabulary)"
        
        similarity_matrix = np.zeros((num_classes, num_classes))
        encoded_labels = self.ov_classifier.cache[self._dataset_name] # retrieve the already normalized encoded labels using CLIP and template ensambling
        similarity_matrix = encoded_labels @ encoded_labels.T

        # Compute weighted sum of similarities
        total_similarity = 0
        total_pixels = np.sum(conf_matrix)
        for gt_class in range(num_classes-1):
            for pred_class in range(num_classes-1):
                total_similarity += conf_matrix[gt_class, pred_class] * similarity_matrix[gt_class, pred_class]

        # Normalize by total pixels to compute mean similarity
        mean_semantic_similarity = total_similarity / total_pixels
        return mean_semantic_similarity.item()
        

    def evaluate(self):
        print("[OVERRIDE] Custom evaluator.evaluate() function")
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal()[:-1].astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix[:-1, :-1], axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix[:-1, :-1], axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        sem_miou = self.label_semantic_metric(self._conf_matrix, self._b_conf_matrix) # Custom metric

        res = {}
        res["mIoU"] = 100 * miou
        res["sem_mIoU"] = 100 * sem_miou # Custom metric result to be added to the pth file
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results