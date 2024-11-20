import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn import metrics
def SaveTensorImage(image, path):
    image = Image.fromarray((np.array(image.cpu()) * 255).astype(np.uint8)).convert("L")
    if path is None:
        image.show()
    else:
        image.save(path)

def CalculateAUC(predict, label, needThreshold = False):

    fpr, tpr, threshold = metrics.roc_curve(label, predict, drop_intermediate=True)  # drop_intermediate=False
    auc = metrics.auc(fpr, tpr)

    return auc, fpr, tpr

def BinaryClassificationMetric(output, label, TP, FP, TN, FN, index):
    output = nn.functional.softmax(output, dim = 1)
    _, predict = torch.max(output, 1)
    P = (predict.int() == 1).int()
    N = (predict.int() == 0).int()
    TP[index] += (P * label.int()).sum().item()
    FP[index] += (P * (1 - label.int())).sum().item()
    TN[index] += (N * (1 - label.int())).sum().item()
    FN[index] += (N * label.int()).sum().item()

def ClassificationMetrics(TP, FP, TN, FN, epsilon = 0):
    accuracy = (TP + TN) / (TP + FP + TN + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    sensitivity = TP / (TP + FN+ epsilon)
    PPV = TP / (TP + FP + epsilon)
    NPV = TN / (TN + FN + epsilon)
    F1 = 2*TP/(2*TP+FP+FN+epsilon)
    return accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1
