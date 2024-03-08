import torch

import scipy
import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
from scipy import ndimage
from scipy.ndimage import convolve, distance_transform_edt as bwdist
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix)

class BMIS_Metrics_Calculator(object):
    def __init__(self, metrics_list):
        super(BMIS_Metrics_Calculator).__init__()

        self.metrics_list = metrics_list

        self.smooth = 1e-5

        self.total_metrics_dict = dict()

        for metric in self.metrics_list:
            self.total_metrics_dict[metric] = list()

    def get_metrics_dict(self, y_pred, y_true):
        y_true = y_true.squeeze().detach().cpu().numpy()
        y_pred = (y_pred.squeeze().detach().cpu().numpy() >= 0.5).astype(np.int_)
        # y_pred = y_pred.squeeze().detach().cpu().numpy()

        y_true = np.asarray(y_true, np.float32)
        y_true /= (y_true.max() + 1e-8)
        y_true[y_true > 0.5] = 1; y_true[y_true != 1] = 0

        metrics_dict = dict()

        for metric in self.metrics_list:
            metrics_dict[metric] = 0
            result = self.get_metrics(metric, y_pred, y_true)
            if np.isnan(result): result = 1e-6
            metrics_dict[metric] = result

        return metrics_dict

    def get_metrics(self, metric, y_pred, y_true):
        if metric == 'Accuracy': return self.calculate_Accuracy(y_pred, y_true)
        elif metric == 'DSC': return self.calculate_DSC(y_pred, y_true)
        elif metric == 'Precision': return self.calculate_Precision(y_pred, y_true)
        elif metric == 'Recall': return self.calculate_Recall(y_pred, y_true)
        elif metric == 'Specificity': return self.calculate_Specificity(y_pred, y_true)
        elif metric == 'Jaccard': return self.calculate_Jaccard(y_pred, y_true)
        elif metric == 'IoU': return self.calculate_IoU(y_pred, y_true)
        elif metric == 'WeightedF-Measure':  return self.calculate_WeightedFMeasure(y_pred, y_true)
        elif metric == 'F-Measure': return self.calculate_FMeasure(y_pred, y_true)
        elif metric == 'S-Measure': return self.calculate_SMeasure(y_pred, y_true)
        elif metric == 'E-Measure': return self.calculate_EMeasure(y_pred, y_true)
        elif metric == 'MAE': return self.calculate_MAE(y_pred, y_true)

    def calculate_Accuracy(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return accuracy_score(y_true, y_pred)

    def calculate_Precision(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return precision_score(y_true, y_pred)

    def calculate_Recall(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return recall_score(y_true, y_pred)

    def calculate_Specificity(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        cm = list(confusion_matrix(y_true, y_pred).ravel())

        if len(cm) == 1: cm += [0, 0, 0]

        tn, fp, fn, tp = cm
        specificity = tn / (tn+fp)

        return specificity

    def calculate_Jaccard(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        return jaccard_score(y_true, y_pred)

    def calculate_DSC(self, y_pred, y_true):
        y_true = np.asarray(y_true.flatten(), dtype=np.int64)
        y_pred = np.asarray(y_pred.flatten(), dtype=np.int64)

        intersection = np.sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / (np.sum(y_true) + np.sum(y_pred) + self.smooth)

    def calculate_IoU(self, y_pred, y_true):
        y_pred_f = y_pred > 0.5
        y_true_f = y_true > 0.5

        intersection_f = (y_pred_f & y_true_f).sum()
        union_f = (y_pred_f | y_true_f).sum()

        iou_f = (intersection_f + self.smooth) / (union_f + self.smooth)

        return iou_f

        # y_pred_b = 1 - y_pred
        # y_true_b = 1 - y_true
        #
        # y_pred_b = y_pred_b > 0.5
        # y_true_b = y_true_b > 0.5
        #
        # intersection_b = (y_pred_b & y_true_b).sum()
        # union_b = (y_pred_b | y_true_b).sum()
        #
        # iou_b = (intersection_b + self.smooth) / (union_b + self.smooth)
        #
        # mIoU = (iou_f + iou_b) / 2.
        #
        # return mIoU


    def calculate_WeightedFMeasure(self, y_pred, y_true):
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(y_true == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(y_pred - y_true)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[y_true == 0] = Et[Idxt[0][y_true == 0], Idxt[1][y_true == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode='constant', cval=0)
        MIN_E_EA = np.where(np.array(y_true, dtype=np.bool_) & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(y_true == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(y_true))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(y_true) - np.sum(Ew[y_true == 1])
        FPw = np.sum(Ew[y_true == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        R = 1 - np.mean(Ew[np.array(y_true, dtype=np.bool_)])
        P = TPw / (1e-6 + TPw + FPw)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = 2 * R * P / (1e-6 + R + P)

        return Q

    def calculate_FMeasure(self, y_pred, y_true):
        th = 2 * y_pred.mean()
        if th > 1:  th = 1
        binary = np.zeros_like(y_pred)
        binary[y_pred >= th] = 1
        hard_gt = np.zeros_like(y_true)
        hard_gt[y_true > 0.5] = 1
        tp = (binary * hard_gt).sum()
        if tp == 0:
            meanF = 0
        else:
            pre = tp / binary.sum()
            rec = tp / hard_gt.sum()
            meanF = 1.3 * pre * rec / (0.3 * pre + rec)

        return meanF

    def calculate_SMeasure(self, y_pred, y_true):
        y = np.mean(y_true)

        if y == 0:
            score = 1 - np.mean(y_pred)
        elif y == 1:
            score = np.mean(y_pred)
        else:
            score = 0.5 * self.object(y_pred, y_true) + 0.5 * self.region(y_pred, y_true)
        return score

    def calculate_EMeasure(self, y_pred, y_true):
        th = 2 * y_pred.mean()
        if th > 1:
            th = 1
        FM = np.zeros(y_true.shape)
        FM[y_pred >= th] = 1
        FM = np.array(FM,dtype=bool)
        GT = np.array(y_true,dtype=bool)
        dFM = np.double(FM)
        if (sum(sum(np.double(GT)))==0):
            enhanced_matrix = 1.0-dFM
        elif (sum(sum(np.double(~GT)))==0):
            enhanced_matrix = dFM
        else:
            dGT = np.double(GT)
            align_matrix = self.AlignmentTerm(dFM, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)
        [w, h] = np.shape(GT)
        score = sum(sum(enhanced_matrix))/ (w * h - 1 + 1e-8)
        return score

    def calculate_MAE(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))

    def matlab_style_gauss2D(self, shape=(7, 7), sigma=5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)

        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        in1 = np.array(in1, dtype=np.int64)
        in2 = np.array(in2, dtype=np.int64)

        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0

        return score

    def AlignmentTerm(self,dFM,dGT):
        mu_FM = np.mean(dFM)
        mu_GT = np.mean(dGT)
        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT
        align_Matrix = 2. * (align_GT * align_FM)/ (align_GT* align_GT + align_FM* align_FM + 1e-8)
        return align_Matrix

    def EnhancedAlignmentTerm(self,align_Matrix):
        enhanced = np.power(align_Matrix + 1,2) / 4
        return enhanced

class Fundus_Image_Segmentation_Metrics_Calculator(object):
    def __init__(self, metrics_list):
        super(Fundus_Image_Segmentation_Metrics_Calculator).__init__()

        self.fundus_label = ['Optic Disk', 'Optic Cup']
        self.metrics_list = metrics_list
        self.smooth = 0.0001
        self.total_metrics_dict = dict()

        for metric in self.metrics_list:
            self.total_metrics_dict[metric] = dict()
            for fundus_label in self.fundus_label:
                self.total_metrics_dict[metric][fundus_label] = list()

    def get_metrics_dict(self, y_pred, y_true):
        y_pred = y_pred.squeeze().cpu().detach().numpy()
        y_true = y_true.squeeze().cpu().detach().numpy()

        metrics_dict = dict()
        for metric in self.metrics_list:
            metrics_dict[metric] = dict()
            for fundus_label in self.fundus_label:
                metrics_dict[metric][fundus_label] = dict()
                metrics_dict[metric][fundus_label] = self.get_metrics(metric, fundus_label, y_pred, y_true)

        return metrics_dict

    def get_metrics(self, metric, fundus_label, y_pred, y_true):
        if fundus_label == 'Optic Cup':
            y_pred = (y_pred[0] >= 0.1).astype(np.uint8)#.flatten()
            y_true = y_true[0]#.flatten()
        elif fundus_label == 'Optic Disk':
            y_pred = (y_pred[1] >= 0.5).astype(np.uint8)#.flatten()
            y_true = y_true[1]#.flatten()

        for i in range(5):
            y_pred = scipy.signal.medfilt2d(y_pred, 7)
        y_pred = morphology.binary_erosion(y_pred, morphology.diamond(7)).astype(np.uint8)
        y_pred = self.get_largest_fillhole(y_pred).astype(np.uint8)  # return 0,1

        if metric == 'IoU': return self.calculate_IoU(y_pred, y_true)
        elif metric == 'DSC': return self.calculate_DSC(y_pred, y_true)
        elif metric == 'Recall': return self.calculate_Recall(y_pred, y_true)

    def calculate_DSC(self, y_pred, y_true):
        '''
        Compute the Dice coefficient between two binary segmentation.
        Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        Input:
            binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
            binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
        Output:
            dice_value: Dice coefficient between the segmentation and the ground truth
        '''
        # turn all variables to booleans, just in case
        binary_segmentation = np.asarray(y_pred, dtype=np.bool_)
        binary_gt_label = np.asarray(y_true, dtype=np.bool_)

        # compute the intersection
        intersection = np.logical_and(binary_segmentation, binary_gt_label)

        # count the number of True pixels in the binary segmentation
        segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
        # same for the ground truth
        gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
        # same for the intersection
        intersection = float(np.sum(intersection.flatten()))

        # compute the Dice coefficient
        dice_value = (2 * intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

        return dice_value

    def calculate_IoU(self, y_pred, y_true):
        '''
        Compute the Dice coefficient between two binary segmentation.
        Dice coefficient is defined as here: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        Input:
            binary_segmentation: binary 2D numpy array representing the region of interest as segmented by the algorithm
            binary_gt_label: binary 2D numpy array representing the region of interest as provided in the database
        Output:
            dice_value: Dice coefficient between the segmentation and the ground truth
        '''
        # turn all variables to booleans, just in case
        binary_segmentation = np.asarray(y_pred, dtype=np.bool_)
        binary_gt_label = np.asarray(y_true, dtype=np.bool_)

        # compute the intersection
        intersection = np.logical_and(binary_segmentation, binary_gt_label)

        # count the number of True pixels in the binary segmentation
        segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
        # same for the ground truth
        gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
        # same for the intersection
        intersection = float(np.sum(intersection.flatten()))

        # compute the Dice coefficient
        iou_value = (intersection + 1.0) / (1.0 + segmentation_pixels + gt_label_pixels)

        return iou_value

    def get_largest_fillhole(self, binary):
        label_image = label(binary)
        regions = regionprops(label_image)
        area_list = []
        for region in regions:
            area_list.append(region.area)
        if area_list:
            idx_max = np.argmax(area_list)
            binary[label_image != idx_max + 1] = 0
        return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def metrics(true, pred) :
    true = true.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pred = (pred >= 0.5).astype(np.int_)

    true = np.asarray(true.flatten(), dtype=np.int64)
    pred = np.asarray(pred.flatten(), dtype=np.int64)

    acc = accuracy_score(true, pred)
    pre = precision_score(true, pred, average='macro')
    rec = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    iou = jaccard_score(true, pred, average='macro')
    cm = confusion_matrix(true, pred).ravel()

    return acc, f1, pre, rec, iou, cm

def MultiClassSegmentationMetrics(true, pred, num_classes):
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for class_idx in range(num_classes):
        TP_list.append(np.sum((true == class_idx) & (pred == class_idx)))
        FP_list.append(np.sum((true != class_idx) & (pred == class_idx)))
        TN_list.append(np.sum((true != class_idx) & (pred != class_idx)))
        FN_list.append(np.sum((true == class_idx) & (pred != class_idx)))

    return TP_list, FP_list, TN_list, FN_list

def Accuracy_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    accuracy_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]
        TN = cfm_per_class['TN'][class_idx]
        FP = cfm_per_class['FP'][class_idx]

        accuracy_per_class.append((TP + TN + eps) / (TN + FP + TP + FN + eps))

    return accuracy_per_class

def Precision_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    precision_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FP = cfm_per_class['FP'][class_idx]

        precision_per_class.append((TP + eps) / (TP + FP + eps))

    return precision_per_class

def Recall_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    recall_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]

        recall_per_class.append((TP + eps) / (TP + FN + eps))

    return recall_per_class

def F1Score_Calculator(precision_per_class, recall_per_class, num_classes=6, eps=0.0001):
    f1_score_per_class = []

    for class_idx in range(num_classes):
        f1_score_per_class.append(2*(precision_per_class[class_idx] * recall_per_class[class_idx] + eps) / (precision_per_class[class_idx] + recall_per_class[class_idx] + eps))

    return f1_score_per_class


def IoU_Calculator(cfm_per_class, num_classes=6, eps=0.0001):
    iou_per_class = []

    for class_idx in range(num_classes):
        TP = cfm_per_class['TP'][class_idx]
        FP = cfm_per_class['FP'][class_idx]
        FN = cfm_per_class['FN'][class_idx]

        iou_per_class.append((TP + eps) / (TP + FP + FN + eps))

    return iou_per_class