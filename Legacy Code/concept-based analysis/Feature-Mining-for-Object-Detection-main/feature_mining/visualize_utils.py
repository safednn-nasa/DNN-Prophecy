import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


from utils.general import non_max_suppression, box_iou
from utils.plots import plot_images, output_to_target


# Version below does not support batches; also will not work with multiple target layers
class Yolov4BoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.
        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, device, nms_conf_thres, nms_iou_thres, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold
        self.device = device
        self.nms_conf_thres=nms_conf_thres
        self.nms_iou_thres=nms_iou_thres

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        output = output.to(self.device, non_blocking=True)

        # Library invokes this function with individual elements of model output tuple
        pred = non_max_suppression(model_outputs, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres)

        # Assuming batch size of 1
        pred = pred[0]

        if len(pred) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            box = box.to(self.device, non_blocking=True)

            iou, index = box_iou(pred[:, :4], box).max(0)  # best ious, indices
            
            if iou > self.iou_threshold and pred[index,5] == label:
                score = iou + pred[index,4]
                output = output + score
        return output


def plot_boxes(save_dir, img, targets, paths, output, names):
    nb, _, height, width = img.shape
    # Plot images
    f = save_dir / f'test_gt_labels.jpg'  # filename
    plot_images(img, targets, paths, f, names)  # labels
    f = os.path.abspath(f)
    img_labels = cv2.imread(f)
    f = save_dir / f'test_pred_labels.jpg'
    plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions
    f = os.path.abspath(f)
    img_pred = cv2.imread(f)
    
    plt.figure(figsize = (20,8))
    tmp = plt.imshow(img_labels)
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize = (20,8))
    tmp = plt.imshow(img_pred)
    plt.axis('off')
    plt.show()