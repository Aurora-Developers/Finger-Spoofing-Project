import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt
import sys
import pandas as pd

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def ConfMat(predicted, actual):
    labels = np.unique(np.concatenate((actual, predicted)))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(actual, predicted):
        true_index = np.where(labels == true_label)[0]
        predicted_index = np.where(labels == predicted_label)[0]
        matrix[predicted_index, true_index] += 1

    return matrix


def OptimalBayes(llr, labels, pi, Cfn, Cfp):
    log_odds = llr
    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    decisions = np.where(log_odds > threshold, 1, 0)

    tp = np.sum(np.logical_and(decisions == 1, labels == 1))
    fp = np.sum(np.logical_and(decisions == 1, labels == 0))
    tn = np.sum(np.logical_and(decisions == 0, labels == 0))
    fn = np.sum(np.logical_and(decisions == 0, labels == 1))

    confusion_matrix = np.array([[tn, fn], [fp, tp]])

    return confusion_matrix


def Bayes_risk(confusion_matrix, pi, Cfn, Cfp):
    M01 = confusion_matrix[0][1]
    M11 = confusion_matrix[1][1]
    M10 = confusion_matrix[1][0]
    M00 = confusion_matrix[0][0]

    FNR = M01 / (M01 + M11)
    FPR = M10 / (M00 + M10)

    DCF = (pi * Cfn * FNR) + ((1 - pi) * Cfp * FPR)

    B = min(pi * Cfn, (1 - pi) * Cfp)

    DCFnorm = DCF / B

    return DCF, DCFnorm


def minCostBayes(llr, labels, pi, Cfn, Cfp):
    if llr.ndim > 1:
        llr = (Cfn * llr[0, :]) / (Cfp * llr[1, :])
    sortedLLR = np.sort(llr)
    # sortedLLR = pi * sortedLLR[0, :] / ((1 - pi) * sortedLLR[1, :])
    t = np.array([-np.inf, np.inf])
    t = np.append(t, sortedLLR)
    t = np.sort(t)
    DCFnorm = []
    FNRlist = []
    FPRlist = []
    for i in t:
        threshold = i
        funct = lambda s: 1 if s > i else 0
        decisions = np.array(list(map(funct, llr)))
        # decisions = np.where(llr > threshold, 1, 0)

        tp = np.sum(np.logical_and(decisions == 1, labels == 1))
        fp = np.sum(np.logical_and(decisions == 1, labels == 0))
        tn = np.sum(np.logical_and(decisions == 0, labels == 0))
        fn = np.sum(np.logical_and(decisions == 0, labels == 1))

        confusion_matrix = np.array([[tn, fn], [fp, tp]])
        M01 = confusion_matrix[0][1]
        M11 = confusion_matrix[1][1]
        M10 = confusion_matrix[1][0]
        M00 = confusion_matrix[0][0]

        FNR = M01 / (M01 + M11)
        FPR = M10 / (M00 + M10)

        [DCF, DCFnormal] = Bayes_risk(confusion_matrix, pi, Cfn, Cfp)

        FNRlist = np.append(FNRlist, FNR)
        FPRlist = np.append(FPRlist, FPR)
        DCFnorm = np.append(DCFnorm, DCFnormal)
    minDCF = min(DCFnorm)

    return minDCF, FPRlist, FNRlist


def ROCcurve(FPRlist, FNRlist):
    TPR = 1 - FNRlist

    plt.figure()
    plt.plot(FPRlist, TPR)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


def BayesErrorPlot(llr, labels, confusion_matrix, Cfn, Cfp):
    DCFlist = []
    minDCFlist = []
    effPriorLogOdds = np.linspace(-6, 6, 21)
    prior = 1 / (1 + np.exp(-effPriorLogOdds))

    for i in prior:
        (_, DCFnorm) = Bayes_risk(confusion_matrix, i, Cfn, Cfp)
        (minDCF, _, _) = minCostBayes(llr, labels, i, Cfn, Cfp)
        DCFlist = np.append(DCFlist, DCFnorm)
        minDCFlist = np.append(minDCFlist, minDCF)

    plt.figure()
    plt.plot(effPriorLogOdds, DCFlist, label="DCF", color="r")
    plt.plot(effPriorLogOdds, minDCFlist, label="min DCF", color="b")
    plt.ylim([0, 3])
    plt.xlim([-1, 6])
    plt.show()


if __name__ == "__main__":
    tableBayes = []
    headers = [
        "(π1, Cfn, Cfp) ",
        "DCFu (B)",
        "Normalized DCF",
        "Min DCF",
        "Normalized DCF e=1",
        "Min DCf e=1",
    ]

    model = "regression"

    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    (train_att, train_label), (test_att, test_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )

    # [SPost, Predictions, accuracy] = ML.Generative_models(
    #     train_att, train_label, test_att, priorProb, test_labels, "tied naive"
    # )

    [_, SPost, accuracy] = ML.binaryRegression(
        train_att, train_label, 0.001, test_att, test_labels
    )

    [Predictions, _] = ML.calculate_model(
        SPost, test_att, "Regression", priorProb, test_labels
    )

    TableCM = ConfMat(Predictions, test_labels)

    # print("Confusion Matrix iris datasets tied covarience", TableCM, sep="\n")

    pi = 0.5
    Cfn = 1
    Cfp = 10
    x = np.argmax(SPost)
    confusion_matrix = OptimalBayes(SPost, test_labels, pi, Cfn, Cfp)
    DCF, DCFnorm = Bayes_risk(TableCM, pi, Cfn, Cfp)
    (minDCF, FPRlist, FNRlist) = minCostBayes(SPost, test_labels, pi, Cfn, Cfp)

    FPR = FPRlist
    FNR = FNRlist
    CM = confusion_matrix

    print(f"Confusion Matrix 1 is:", f" {confusion_matrix}", sep="\n")

    listVals = [DCF, DCFnorm, minDCF]
    tableBayes.append([f"({pi},{Cfn},{Cfp})"])

    for i in listVals:
        tableBayes[0].append(i)

    print(tabulate(tableBayes, headers=headers))

    ROCcurve(FPR, FNR)
    BayesErrorPlot(SPost, test_labels, TableCM, Cfn, Cfp)
