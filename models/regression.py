import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

tablePCA = []
tableKFold = []
headers = ["Dimensions", "Logistic Regression"]
pi = 0.5
Cfn = 1
Cfp = 10


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def split_db(D, L, fraction, seed=0):
    nTrain = int(D.shape[1] * fraction)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    l = float(input("Lambda parameter: "))
    ### ------------- PCA WITH 2/3 SPLIT ---------------------- ####
    (train_att, train_label), (test_att, test_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )

    tablePCA.append(["Full"])

    [Predictions, SPost, accuracy] = ML.binaryRegression(
        train_att, train_label, 0.001, test_att, test_labels
    )

    confusion_matrix = ML.ConfMat(Predictions, test_labels)
    DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    (minDCF, _, _) = ML.minCostBayes(SPost, test_labels, pi, Cfn, Cfp)
    tablePCA[0].append([accuracy, DCFnorm, minDCF])

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)

        tablePCA.append([f"PCA {i}"])

        [_, SPost, accuracy] = ML.binaryRegression(
            reduced_train, train_label, l, reduced_test, test_labels
        )
        [Predictions, _] = ML.calculate_model(
            SPost, test_att, "Regression", priorProb, test_labels
        )
        confusion_matrix = ML.ConfMat(Predictions, test_labels)
        DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
        (minDCF, _, _) = ML.minCostBayes(SPost, test_labels, pi, Cfn, Cfp)
        tablePCA[cont].append([accuracy, DCFnorm, minDCF])
        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            tablePCA.append([f"PCA {i} LDA {j}"])
            W, _ = ML.LDA1(reduced_train, train_label, j)
            LDA_train = np.dot(W.T, reduced_train)
            LDA_test = np.dot(W.T, reduced_test)
            [_, SPost, accuracy] = ML.binaryRegression(
                LDA_train, train_label, l, LDA_test, test_labels
            )
            [Predictions, _] = ML.calculate_model(
                SPost, test_att, "Regression", priorProb, test_labels
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
            (minDCF, _, _) = ML.minCostBayes(SPost, test_labels, pi, Cfn, Cfp)
            tablePCA[cont].append([accuracy, DCFnorm, minDCF])
            cont += 1

    print("PCA with a 2/3 split")
    print(tabulate(tablePCA, headers=headers))

    ### K-fold binomial Regression ###

    tableKFold.append(["Full"])

    print(f"Size of dataset: {full_train_att.shape[1]}")
    k = int(input("Number of partitions: "))

    [_, _, accuracy, DCFNorm, minDCF] = ML.k_fold(
        k, full_train_att, full_train_label, priorProb, "regression", l=l
    )
    tableKFold[0].append([accuracy, DCFnorm, minDCF])

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break

        tableKFold.append([f"PCA {i}"])
        [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
            k, full_train_att, full_train_label, priorProb, "regression", PCA_m=i, l=l
        )
        tableKFold[cont].append([accuracy, DCFnorm, minDCF])

        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            tableKFold.append([f"PCA {i} LDA {j}"])
            [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k,
                full_train_att,
                full_train_label,
                priorProb,
                "regression",
                PCA_m=i,
                LDA_m=j,
                l=l,
            )
            tableKFold[cont].append([accuracy, DCFnorm, minDCF])
            cont += 1

    print("PCA with k-fold")
    print(tabulate(tableKFold, headers=headers))
