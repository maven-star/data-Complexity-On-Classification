import Proposed_Data_complexity_aware_RideNN_ensemble_classifier.Ridder_NN
from scipy.spatial import distance
import numpy as np

def finding_highest_prob(prob):
    max_index = 0
    list_len = len(prob)
    for index in range(list_len):
        if prob[index] > prob[max_index]:
            max_index = index
    return max_index

def callmain(Cgroup,Clabel,cs,tr,ACC,TPR,TNR):
    prob = []
    OUT, YTest = [], []
    for i in range(cs):
        out, ytest = Proposed_Data_complexity_aware_RideNN_ensemble_classifier.Ridder_NN.Classify(Cgroup[i], Clabel[i], tr)
        OUT.append(out)
        YTest.append(ytest)
        dist = distance.dice(out, ytest)
        prob.append(dist)
    indx = finding_highest_prob(prob)

    out = OUT[indx]
    target = YTest[indx]

    tp, tn, fn, fp = 1, 1, 1, 1
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if target[i] == c and out[i] == c:
                fp += 1
            if target[i] != c and out[i] != c:
                tn += 1
            if (target[i] == c and out[i] != c):
                tp += 1
            if (target[i] != c and out[i] == c):
                fn += 1

    acc = (tp + tn) / (tp + fn + tn + fp)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ACC.append(acc)
    TPR.append(tpr)
    TNR.append(tnr)