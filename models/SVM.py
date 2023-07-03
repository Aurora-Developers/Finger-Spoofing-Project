import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

pi = 0.5
Cfn = 1
Cfp = 10
tablePCA = []
tableKFold = []
# headers = [
#     "SVM full:[0.1, 1]",
#     "SVM full:[0.1, 10]",
#     "SVM full:[1, 1]",
#     "SVM full:[1, 10]",
#     "Polynomial:[0.1, 0, 2, 0]",
#     "Polynomial:[0.1, 1, 2, 1]",
#     "Polynomial:[1, 0, 2, 0]",
#     "Polynomial:[1, 1, 2, 1]",
#     "Polynomial:[0.1, 1, 0.01]",
#     "Polynomial:[0.1, 1, 0.01]",
#     "Radial:[0.1, 1]",
#     "Radial:[0.1, 10]",
#     "Radial:[1, 1]",
#     "Radial:[1, 10]",
# ]

headers = [
    "SVM full:[0.1, 1]",
    "SVM full:[0.1, 10]",
    "SVM full:[1, 1]",
    "SVM full:[1, 10]",
]


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

    cont = 0
    c = 0
    initial_C = 0.1
    initial_K = 1
    total_iter = 36

    tablePCA.append(["Full"])

    for ten in range(2):
        contrain = initial_C * np.power(10, ten)
        for j in range(2):
            k = initial_K * np.power(10, j)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
                SPost, test_labels, pi, Cfn, Cfp
            )
            tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])
            print(f"{round(cont * 100 / total_iter, 2)}%")
            cont += 1

    # initial_d = 2
    # initial_const = 0
    # initial_K = 0
    # for i in range(2):
    #     contrain = initial_C * np.power(10, i)
    #     d = initial_d
    #     for j in range(2):
    #         const = initial_const + j
    #         k = initial_K + j
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att,
    #             train_label,
    #             test_att,
    #             test_labels,
    #             contrain,
    #             dim=d,
    #             c=const,
    #             eps=k**2,
    #             model="polynomial",
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
    #             SPost, test_labels, pi, Cfn, Cfp
    #         )
    #         tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])
    #         print(f"{round(cont * 100 / total_iter, 2)}%")
    #         cont += 1
    # initial_gamma = 1
    # for ten in range(2):
    #     contrain = initial_C * np.power(10, ten)
    #     for j in range(2):
    #         gamma = initial_gamma * np.power(10, j)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att,
    #             train_label,
    #             test_att,
    #             test_labels,
    #             contrain,
    #             gamma=gamma,
    #             model="radial",
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
    #             SPost, test_labels, pi, Cfn, Cfp
    #         )
    #         tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])
    #         print(f"{round(cont * 100 / total_iter, 2)}%")
    #         cont += 1
    cont += 1
    c += 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)
        tablePCA.append([f"PCA {i}"])
        initial_K = 1
        for ten in range(2):
            contrain = initial_C * np.power(10, ten)
            for j in range(3):
                k = initial_K * np.power(10, j)
                [SPost, Predictions, accuracy] = ML.svm(
                    reduced_train, train_label, reduced_test, test_labels, contrain, K=k
                )
                confusion_matrix = ML.ConfMat(Predictions, test_labels)
                DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
                (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
                    SPost, test_labels, pi, Cfn, Cfp
                )
                tablePCA[c].append([round(accuracy * 100, 2), DCFnorm, minDCF])
                print(f"{round(cont * 100 / total_iter, 2)}%")
                cont += 1
        # initial_K = 0
        # for i in range(2):
        #     contrain = initial_C * np.power(10, i)
        #     d = initial_d
        #     for j in range(2):
        #         const = initial_const + j
        #         k = initial_K + j
        #         [SPost, Predictions, accuracy] = ML.svm(
        #             reduced_train,
        #             train_label,
        #             reduced_test,
        #             test_labels,
        #             contrain,
        #             dim=d,
        #             c=const,
        #             eps=k**2,
        #             model="polynomial",
        #         )
        #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
        #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
        #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
        #             SPost, test_labels, pi, Cfn, Cfp
        #         )
        #         tablePCA[c].append([round(accuracy * 100, 2), DCFnorm, minDCF])
        #         print(f"{round(cont * 100 / total_iter, 2)}%")
        #         cont += 1
        # for ten in range(2):
        #     contrain = initial_C * np.power(10, ten)
        #     for j in range(2):
        #         gamma = initial_gamma * np.power(10, j)
        #         [SPost, Predictions, accuracy] = ML.svm(
        #             reduced_train,
        #             train_label,
        #             reduced_test,
        #             test_labels,
        #             contrain,
        #             gamma=gamma,
        #             model="radial",
        #         )
        #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
        #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
        #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
        #             SPost, test_labels, pi, Cfn, Cfp
        #         )
        #         tablePCA[c].append([round(accuracy * 100, 2), DCFnorm, minDCF])
        #         print(f"{round(cont * 100 / total_iter, 2)}%")
        #         cont += 1
        c += 1

    print("PCA with a 2/3 split")
    print(tabulate(tablePCA, headers=headers))
