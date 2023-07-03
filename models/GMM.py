import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

tableGMM = []
tableDiag = []
tableTied = []
headers = [
    "GMM:[0.1, 0, 0.01]",
    "GMM:[0.1, 1, 0.01]",
    "GMM:[0.1, 2, 0.01]",
    "Diagonal:[0.1, 0, 0.01]",
    "Diagonal:[0.1, 1, 0.01]",
    "Diagonal:[0.1, 2, 0.01]",
    "Tied:[0.1, 0, 0.01]",
    "Tied:[0.1, 1, 0.01]",
    "Tied:[0.1, 2, 0.01]",
]
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

    total_iter = 37 * len(headers)
    perc = 0

    priorProb = ML.vcol(np.ones(2) * 0.5)
    psi = 0.01
    ### ------------- PCA WITH 2/3 SPLIT ---------------------- ####
    (train_att, train_label), (test_att, test_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )
    tableGMM.append(["Full"])
    tableDiag.append(["Full"])
    tableTied.append(["Full"])

    for model in headers:
        ref = model.split(":")
        mod = ref[0]
        constrains = eval(ref[1])

        [SPost, Predictions, accuracy] = ML.Generative_models(
            train_att,
            train_label,
            test_att,
            priorProb,
            test_labels,
            mod,
            niter=constrains[1],
            alpha=constrains[0],
            psi=constrains[2],
        )
        confusion_matrix = ML.ConfMat(Predictions, test_labels)
        DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
        (minDCF, FPRlist, FNRlist) = ML.minCostBayes(SPost, test_labels, pi, Cfn, Cfp)
        if mod == "GMM":
            tableGMM[0].append([accuracy, DCFnorm, minDCF])
        elif mod == "Diagonal":
            tableDiag[0].append([accuracy, DCFnorm, minDCF])
        elif mod == "Tied":
            tableTied[0].append([accuracy, DCFnorm, minDCF])
        perc += 1
        print(f"{round(perc * 100 / total_iter, 2)}%")
    print()
    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)

        tableGMM.append([f"PCA {i}"])
        tableDiag.append([f"PCA {i}"])
        tableTied.append([f"PCA {i}"])
        for model in headers:
            ref = model.split(":")
            mod = ref[0]
            constrains = eval(ref[1])

            [SPost, Predictions, accuracy] = ML.Generative_models(
                reduced_train,
                train_label,
                reduced_test,
                priorProb,
                test_labels,
                mod,
                niter=constrains[1],
                alpha=constrains[0],
                psi=constrains[2],
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
                SPost, test_labels, pi, Cfn, Cfp
            )
            if mod == "GMM":
                tableGMM[cont].append([accuracy, DCFnorm, minDCF])
            elif mod == "Diagonal":
                tableDiag[cont].append([accuracy, DCFnorm, minDCF])
            elif mod == "Tied":
                tableTied[cont].append([accuracy, DCFnorm, minDCF])
            perc += 1
            print(f"{round(perc * 100 / total_iter, 2)}%", end="\t")
        print()
        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            tableGMM.append([f"PCA {i} LDA {j}"])
            tableDiag.append([f"PCA {i} LDA {j}"])
            tableTied.append([f"PCA {i} LDA {j}"])
            W, _ = ML.LDA1(reduced_train, train_label, j)
            LDA_train = np.dot(W.T, reduced_train)
            LDA_test = np.dot(W.T, reduced_test)
            for model in headers:
                ref = model.split(":")
                mod = ref[0]
                constrains = eval(ref[1])

                [SPost, Predictions, accuracy] = ML.Generative_models(
                    LDA_train,
                    train_label,
                    LDA_test,
                    priorProb,
                    test_labels,
                    mod,
                    niter=constrains[1],
                    alpha=constrains[0],
                    psi=constrains[2],
                )
                confusion_matrix = ML.ConfMat(Predictions, test_labels)
                DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
                (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
                    SPost, test_labels, pi, Cfn, Cfp
                )
                if mod == "GMM":
                    tableGMM[cont].append([accuracy, DCFnorm, minDCF])
                elif mod == "Diagonal":
                    tableDiag[cont].append([accuracy, DCFnorm, minDCF])
                elif mod == "Tied":
                    tableTied[cont].append([accuracy, DCFnorm, minDCF])
                perc += 1
                print(f"{round(perc * 100 / total_iter, 2)}%", end="\t")
            print()
            cont += 1
    # headersPCA = []
    # for i in headers:
    #     headersPCA.append(i + " Acc/DCF/MinDCF")
    print("PCA GMM with a 2/3 split")
    print(tabulate(tableGMM, headers=headers[0:3]))
    print("PCA Diagonal with a 2/3 split")
    print(tabulate(tableDiag, headers=headers[3:6]))
    print("PCA Tied with a 2/3 split")
    print(tabulate(tableTied, headers=headers[6:]))

    ### ------------- k-fold with different PCA ---------------------- ####
    # headers = ["MVG", "Naive", "Tied Gaussian", "Tied Naive"]
    # tableKFold.append(["Full"])
    # print(f"Size of dataset: {full_train_att.shape[1]}")
    # # k_fold_value = int(input("Value for k partitions: "))
    # k_fold_value = 20
    # for model in headers:
    #     [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
    #         k_fold_value, full_train_att, full_train_label, priorProb, model
    #     )
    #     tableKFold[0].append([accuracy, DCFnorm, minDCF])

    # cont = 1
    # for i in reversed(range(10)):
    #     if i < 2:
    #         break

    #     tableKFold.append([f"PCA {i}"])
    #     for model in headers:
    #         [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
    #             k_fold_value,
    #             full_train_att,
    #             full_train_label,
    #             priorProb,
    #             model,
    #             PCA_m=i,
    #         )
    #         tableKFold[cont].append([accuracy, DCFnorm, minDCF])

    #     cont += 1
    #     for j in reversed(range(i)):
    #         if j < 2:
    #             break
    #         tableKFold.append([f"PCA {i} LDA {j}"])
    #         for model in headers:
    #             [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
    #                 k_fold_value,
    #                 full_train_att,
    #                 full_train_label,
    #                 priorProb,
    #                 model,
    #                 PCA_m=i,
    #                 LDA_m=j,
    #             )
    #             tableKFold[cont].append([accuracy, DCFnorm, minDCF])

    #         cont += 1

    # print("PCA with k-fold")
    # print(tabulate(tableKFold, headers=headers))
