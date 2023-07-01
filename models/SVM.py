import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

tablePCA = []
tableKFold = []
headers = ["Dataset/Hyperparameter", "Method / Accuracy"]


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

    tablePCA.append(["Full"])
    tablePCA[0].append(["Dual SVM"])

    cont = 0
    initial_C = 0.01
    initial_K = 1
    total_iter = 24

    for ten in range(3):
        contrain = initial_C * np.power(10, ten)
        for j in range(3):
            k = initial_K * np.power(10, j)
            tablePCA.append([f"C: {contrain}, K: {k}"])
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k
            )
            cont += 1
            tablePCA[cont].append(round(accuracy * 100, 2))
            print(f"{round(cont * 100 / total_iter, 2)}%")
    cont += 1
    tablePCA.append(["Full"])
    tablePCA[cont].append(["Polynomial"])
    initial_d = 2
    initial_const = 0
    for i in range(3):
        contrain = initial_C * np.power(10, i)
        d = initial_d + i
        for j in range(2):
            const = initial_const + j
            tablePCA.append([f"C: {contrain} d: {d}, c: {const}"])
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                dim=d,
                c=const,
                model="polynomial",
            )
            cont += 1
            tablePCA[cont].append(round(accuracy * 100, 2))
            print(f"{round(cont * 100 / total_iter, 2)}%")
    cont += 1
    tablePCA.append(["Full"])
    tablePCA[cont].append(["Radial"])
    initial_gamma = 1
    for ten in range(3):
        contrain = initial_C * np.power(10, ten)
        for j in range(3):
            gamma = initial_gamma + j
            tablePCA.append([f"C: {contrain}, gamma: {gamma}"])
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                gamma=gamma,
                model="radial",
            )
            cont += 1
            tablePCA[cont].append(round(accuracy * 100, 2))
            print(f"{round(cont * 100 / total_iter, 2)}%")

    # cont = 1
    # for i in reversed(range(10)):
    #     if i < 2:
    #         break
    #     P, reduced_train = ML.PCA(train_att, i)
    #     reduced_test = np.dot(P.T, test_att)

    #     tablePCA.append([f"PCA {i}"])

    #     [modelS, _, accuracy] = ML.binaryRegression(
    #         reduced_train, train_label, l, reduced_test, test_labels
    #     )
    #     tablePCA[cont].append(accuracy)
    #     cont += 1
    #     for j in reversed(range(i)):
    #         if j < 2:
    #             break
    #         tablePCA.append([f"PCA {i} LDA {j}"])
    #         W, _ = ML.LDA1(reduced_train, train_label, j)
    #         LDA_train = np.dot(W.T, reduced_train)
    #         LDA_test = np.dot(W.T, reduced_test)
    #         [modelS, _, accuracy] = ML.binaryRegression(
    #             LDA_train, train_label, l, LDA_test, test_labels
    #         )
    #         tablePCA[cont].append(accuracy)
    #         cont += 1

    print("PCA with a 2/3 split")
    print(tabulate(tablePCA, headers=headers))
