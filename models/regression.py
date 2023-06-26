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

    ### ------------- PCA WITH 2/3 SPLIT ---------------------- ####
    (train_att, train_label), (test_att, test_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )

    l = 0.001

    tablePCA.append(["Full"])

    [modelS, _, accuracy] = ML.binaryRegression(
        train_att, train_label, l, test_att, test_labels
    )
    tablePCA[0].append(accuracy)

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)

        tablePCA.append([f"PCA {i}"])

        [modelS, _, accuracy] = ML.binaryRegression(
            reduced_train, train_label, l, reduced_test, test_labels
        )
        tablePCA[cont].append(accuracy)
        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            tablePCA.append([f"PCA {i} LDA {j}"])
            W, _ = ML.LDA1(reduced_train, train_label, j)
            LDA_train = np.dot(W.T, reduced_train)
            LDA_test = np.dot(W.T, reduced_test)
            [modelS, _, accuracy] = ML.binaryRegression(
                LDA_train, train_label, l, LDA_test, test_labels
            )
            tablePCA[cont].append(accuracy)
            cont += 1

    print("PCA with a 2/3 split")
    print(tabulate(tablePCA, headers=headers))
