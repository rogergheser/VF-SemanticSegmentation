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
    def __init__(self, model, dataset_name, distributed, output_dir=None):
        super().__init__(dataset_name, distributed, output_dir)
        self.ov_classifier = model.ov_classifier # san linear classifier to project the pixel embedding into the vocabulary space
        self.inference_voc = []
        self.word_to_gt = {}

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"]
            if output.shape[0] > 1:
                output = output.argmax(dim=0).to(self._cpu_device)
            else:
                output = output.squeeze(0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            gt[gt == self._ignore_label] = self._num_classes

            if "vocabulary" in input:
                if not self.inference_voc:
                    self.inference_voc = {voc: i+self._num_classes+1 for i, voc in enumerate(input["vocabulary"])}
                else:
                    for word in self.inference_voc:
                        if word not in input["vocabulary"]:
                            self.inference_voc[word] = self._num_classes + 1 + len(self.inference_voc)

                for i, word in enumerate(input["vocabulary"]):
                    pred[pred == i] = self.inference_voc[word]

                    if len(pred[pred == self.inference_voc[word]]) > 0:
                        # Count the number of pixels assigned to each GT label for the current word
                        counts = np.bincount(
                            gt[pred == self.inference_voc[word]], minlength=self._num_classes+1
                        )
                        
                        # Store the counts in the dictionary
                        self.word_to_gt[word] = counts[:-1] # remove the last element of the counts (ignore label)
                    pred[pred == self.inference_voc[word]] = self._num_classes # Assign the ignore label to the pixels assigned to the word outside the gt vocabulary

            
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))


    def pixel_semantics_metric(self):
        """
        Custom metric that computes the mean semantic pixel similarity between the predicted class and the ground truth class.
        Semantic similarity between two classes is defined as dot product of the encoded class vectors.

        Returns:
            float: mean semantic pixel similarity
        """
        if self._dataset_name not in self.ov_classifier.cache:
            self.ov_classifier.get_classifier_by_dataset_name(self._dataset_name) # populate the cache with the encoded labels
        
        # if self.inference_voc:
        #     assert self._num_classes + len(self.inference_voc) + 1 == self._conf_matrix.shape[0], "The number of classes in the inference vocabulary must match the number of classes in the dataset."
        # else:
        #     assert self._num_classes + 1 == self._conf_matrix.shape[0], "The number of classes in the vocabulary must match the number of classes in the dataset."

        num_classes = self._conf_matrix.shape[0] - 1
        similarity_matrix = np.zeros((num_classes, num_classes))
        encoded_labels: torch.Tensor = self.ov_classifier.cache[self._dataset_name] # retrieve the already normalized encoded labels using CLIP and template ensambling
        if self.inference_voc:
            encoded_labels = torch.cat((self.ov_classifier.get_classifier_by_vocabulary(self.inference_voc), encoded_labels), dim=0)
        similarity_matrix: torch.Tensor = encoded_labels @ encoded_labels.T

        # rescacle the similarity matrix to [0, 1]
        sim_min = similarity_matrix.min()
        sim_max = similarity_matrix.max()
        similarity_matrix_rescaled = (similarity_matrix - sim_min) / (sim_max - sim_min)

        conf_matrix = self._conf_matrix[:-1, :-1] # remove the last row and column of the confusion matrix (ignore label)
        if self.inference_voc:
            for word in self.inference_voc:
                #add a row to conf_matrix for the current word
                if word in self.word_to_gt:
                    conf_matrix = np.vstack((conf_matrix, self.word_to_gt[word])) # remove the last element of the counts (ignore label)
            print(conf_matrix.shape)

        # Compute weighted sum of similarities
        total_similarity = 0
        total_similarity_rescaled = 0
        total_pixels = np.sum(conf_matrix)
        for pred_class in range(conf_matrix.shape[0]):
            for gt_class in range(conf_matrix.shape[1]):
                total_similarity += conf_matrix[pred_class, gt_class] * similarity_matrix[pred_class, gt_class]
                total_similarity_rescaled += conf_matrix[pred_class, gt_class] * similarity_matrix_rescaled[pred_class, gt_class]

        # Normalize by total pixels to compute mean similarity
        mean_semantic_similarity = total_similarity / total_pixels
        mean_semantic_similarity_rescaled = total_similarity_rescaled / total_pixels
        return mean_semantic_similarity.item(), mean_semantic_similarity_rescaled.item()
        

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

        # if not self.inference_voc:
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

        sem_acc, sem_acc_rescaled = self.pixel_semantics_metric() # Custom metric

        res = {}
        res["conf_matrix"] = self._conf_matrix # Saves also the confusion matrix for future "analysis
        res["word_to_gt"] = self.word_to_gt # Saves the number of pixels assigned to each GT label for each word in the inference vocabulary
        res["sem_acc"] = 100 * sem_acc # Custom metric result to be added to the pth file
        res["sem_acc_rescaled"] = 100 * sem_acc_rescaled
        print(f"Semantic pixel similarity: {res['sem_acc']:.2f}%")

        if not self.inference_voc:
            res["mIoU"] = 100 * miou
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
        del res["conf_matrix"] # Remove the confusion matrix from the results to avoid print error in the logger
        del res["word_to_gt"] # Remove the word_to_gt from the results to avoid print error in the logger
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
    
    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []
