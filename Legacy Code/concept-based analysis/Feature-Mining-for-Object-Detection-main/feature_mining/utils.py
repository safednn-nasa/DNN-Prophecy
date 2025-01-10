from os import stat
from typing import List

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.general import box_iou, xyxy2xywh, xywh2xyxy, clip_coords, non_max_suppression
from utils.metrics import ap_per_class, compute_ap



""" Use as follows:
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(testloader)):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        with torch.no_grad():
            pred = model(img)  # forward
            output = non_max_suppression(pred[0], conf_thres=conf_thres, iou_thres=iou_thres)
            assert len(output)==1, "batch size should be 1"
            img = torch.squeeze(img)
            output = output[0]
            labels = targets[targets[:, 0] == 0, 1:]
            mAP = mAP_per_image(img, output, labels, nc, device)

"""
def mAP_per_image(img, pred, labels, nc, device):
    stats = []
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    _, height, width = img.shape  # channels, height, width
    whwh = torch.Tensor([width, height, width, height]).to(device)
    nl = len(labels)
    tcls = labels[:, 0].tolist() if nl else []  # target class

    if len(pred) == 0:
        if nl:
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
    else:
        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5]) * whwh

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        return map50
    else:
        # case when either no true positive prediction or no targets
        nt = torch.zeros(1)
        return 0.0

def accumulate_stats(stats : List, model_output: torch.Tensor, targets: torch.Tensor, nc, device: torch.device, height: int, width: int, attributes_gt=None, nattributes: int = 0):
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    whwh = torch.Tensor([width, height, width, height]).to(device)

    for si, pred in enumerate(model_output):
        labels = targets[targets[:, 0] == si, 1:]
        if attributes_gt is not None:
            image_attributes = attributes_gt[targets[:, 0] == si, 1:]
        else:
            image_attributes = None
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        else:
            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            # Related attributes
            pred_attribute = torch.zeros(pred.shape[0], nattributes, dtype=torch.bool, device=device) # shape is pred.shape[0] x # of attributes
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor, sorted = False):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if attributes_gt is not None and image_attributes is not None:
                                    d_attri = (image_attributes[ti[i[j]]] > 0) # Cast float to bool
                                    pred_attribute[pi[j]] = d_attri
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            if attributes_gt is None or image_attributes is None:
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            else:
                # TODO: append attribute to the stats
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls, pred_attribute.cpu(), image_attributes.cpu()))
    return


def stats_dataset(nuim, data_root, model, dataloader, device, names, attribute_names, nc, conf_thres, iou_thres):

    attribute_token_to_idx = {}
    for i, attribute in enumerate(nuim.attribute):
        attribute_token_to_idx.update({attribute['token'] : i})

    category_token_to_idx = {}
    for i, category in enumerate(nuim.category):
        category_token_to_idx.update({category['token'] : i})

    filename_to_attributes = {}    # {filename -> {category_token -> attribute_idxs}}
    for ob in nuim.object_ann:
        sample_data_token = ob['sample_data_token']
        if ob['mask'] is None:
            continue
        category = category_token_to_idx[ob['category_token']]
        sample = nuim.get('sample_data', sample_data_token)
        filename = data_root + sample['filename']
        attribute = [attribute_token_to_idx[x] for x in ob['attribute_tokens']]
        if filename in filename_to_attributes:
            if category in filename_to_attributes[filename]:
                filename_to_attributes[filename][category].append(attribute)
            else:
                filename_to_attributes[filename].update({category : [attribute]})
        else:
            filename_to_attributes.update({filename : {category : [attribute]}})
    
    nattributes = len(nuim.attribute)

    stats = []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # targets: idx, class idx, x, y, x, y
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        # Initialize the attribute ground truth
        attributes_gt = torch.zeros(targets.shape[0], nattributes + 1, device=device)
        # Initialize the index column
        attributes_gt[:, 0] = targets[:, 0]
        for i, filename in enumerate(paths):
            if filename in filename_to_attributes:
                tmp = torch.zeros((targets[:, 0] == i).sum(), nattributes, device=device)
                j = 0
                # print(filename_to_attributes[filename])
                for cate in filename_to_attributes[filename].keys():
                    for obj in filename_to_attributes[filename][cate]:
                        tmp[j, obj] = 1
                        j += 1
                # print(j, (targets[:, 0] == i).sum())
                assert j == tmp.shape[0]
                # print(tmp)
                attributes_gt[attributes_gt[:, 0] == i, 1:] = tmp
        # print(attributes_gt)

        with torch.no_grad():
            inf_out = model(img)[0]
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)
            accumulate_stats(stats, output, targets, nc, device, height, width, attributes_gt, nattributes)
    raw_stats = stats
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    p, r, ap, f1, ap_class, r_attribute, ap_attributes = ap_per_class_attribute(*stats, plot=False)

    # Print the results
    print('{:<50}\t{:<40}\t{:<10}'.format('Class', 'Attribute', 'Recall@0.1'))
    for i, cls in enumerate(ap_class):
        for j, attri in enumerate(ap_attributes):
            if r_attribute[i, j, 0] != -1:
                print('{:<50}\t{:<40}\t{:.10f}'.format(names[cls], attribute_names[attri], r_attribute[i, j, 0]))

    print()
    
    print('{:<36}\t{:<10}\t{:<10}\t{:<10}'.format('Class', 'Pricision@0.1', 'Recall@0.1', 'mAP for class'))
    for i, cls in enumerate(ap_class):
        print('{:<36}\t{:.10f}\t{:.10f}\t{:.10f}'.format(names[cls], p[i][0], r[i][0], np.mean(ap[i])) )

    print()

    print('{:<36}\t{:<10}'.format('Class', 'AP@0.5:0.95'))
    for i, cls in enumerate(ap_class):
        print('{:<36}\t'.format(names[cls]), end='')
        for j in ap[i]:
            print('{:5f}  '.format(j), end='')
        print()

    print()
    print('{:<10}\t{:<10}\t{:<10}'.format('Mean Pricision', 'Mean Recall', 'mAP'))
    print('{:.10f}\t{:.10f}\t{:.10f}'.format(np.mean(p[:,0]), np.mean(r[:,0]), np.mean(ap)))
    return p, r, ap, f1, ap_class, r_attribute, ap_attributes, raw_stats


def ap_per_class_attribute(tp, conf, pred_cls, target_cls, pred_attribute, target_attribute, plot=False, fname='precision-recall_curve.png'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        fname:  Plot filename
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls, pred_attribute = tp[i], conf[i], pred_cls[i], pred_attribute[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    
    # Number of unique attributes in the dataset
    num_unique_attributes = np.unique(target_attribute.nonzero()[1]).shape[0]

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    s_r = [unique_classes.shape[0], num_unique_attributes, tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    r_attribute = -np.ones(s_r)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_p = i.sum()  # number of predictions
        n_l = (target_cls == c).sum()  # number of labels
        fpc = (1 - tp[i]).cumsum(0)

        if n_p == 0 or n_l == 0:
            continue

        attributes = pred_attribute[i]
        # Find unique attributes
        attri_idx = target_cls == c
        unique_attributes = np.unique(target_attribute[attri_idx].nonzero()[1])

        for ai, a in enumerate(unique_attributes):
            n_l_attri = target_attribute[attri_idx, a].sum() #look at both target_cls and gt_attribute to get target objects with class c and attribute a

            if n_p == 0 or n_l_attri == 0:
                continue

            idx1 = i #indices of "correctly" labeled predicted objects with class c
            tpc1 = tp[i]
            idx2 = attributes[:, a] == 1 # indices filtered from idx1 which have attribute a
            tpc = tpc1[idx2].cumsum(0)

            # Recall
            recall = tpc / (n_l_attri + 1e-16)  # recall curve
            if conf[i][idx2].shape[0] > 0 and recall[:, 0].shape[0] > 0:
                r_attribute[ci, a] = np.interp(-pr_score, -conf[i][idx2], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases
                # print(r_attribute[ci, a])
            else:
                r_attribute[ci, a] = 0
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + 1e-16)  # recall curve
        r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    if plot:
        py = np.stack(py, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(px, py, linewidth=0.5, color='grey')  # plot(recall, precision)
        ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fname, dpi=200)

    return p, r, ap, f1, unique_classes.astype('int32'), r_attribute, np.unique(target_attribute.nonzero()[1]).astype('int32')