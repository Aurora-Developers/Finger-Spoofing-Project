import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

tablePCA = []
tableKFold = []
headers = ["Dimensions", "Logistic Regression ACC/DCF/minDCF"]
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
    l_list = np.linspace(-(10**-5), 1, num=5)
    minDCf7_list = []
    minDCffull_list = []
    minDCf8_list = []
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    # ### ------------- PCA WITH 2/3 SPLIT ---------------------- ####

    for l in l_list:
        tablePCA = []
        (train_att, train_label), (test_att, test_labels) = ML.split_db(
            full_train_att, full_train_label, 2 / 3
        )

        tablePCA.append(["Full"])

        [Predictions, SPost, accuracy] = ML.binaryRegression(
            train_att, train_label, l, test_att, test_labels
        )

        confusion_matrix = ML.ConfMat(Predictions, test_labels)
        DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
        (minDCF, _, _) = ML.minCostBayes(SPost, test_labels, pi, Cfn, Cfp)
        minDCffull_list.append(minDCF)
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

            if i == 7:
                minDCf7_list.append(minDCF)
            if i == 8:
                minDCf8_list.append(minDCF)

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

        print(f"PCA with a 2/3 split with lambda={l}")
        print(tabulate(tablePCA, headers=headers))
    print(minDCf7_list)
    print(minDCffull_list)
    print(minDCf8_list)
    plt.figure()
    # x = np.linspace(0, 10, num=100)
    # y1 = np.sin(x)
    # y2 = np.cos(x)

    # # Plot the arrays
    plt.plot(np.log(l_list), minDCf7_list)
    plt.plot(np.log(l_list), minDCf8_list)
    plt.plot(np.log(l_list), minDCffull_list)

    # # Add labels and a legend
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()

    # # Show the plot
    plt.show()

    ### K-fold binomial Regression ###
    for l in l_list:
        tableKFold.append(["Full"])

        print(f"Size of dataset: {full_train_att.shape[1]}")
        k = int(input("Number of partitions: "))

        [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
            k, full_train_att, full_train_label, priorProb, model="regression", l=l
        )
        tableKFold[0].append([accuracy, DCFnorm, minDCF])

        cont = 1
        for i in reversed(range(10)):
            if i < 2:
                break

            tableKFold.append([f"PCA {i}"])
            [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k,
                full_train_att,
                full_train_label,
                priorProb,
                model="regression",
                PCA_m=i,
                l=l,
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
                    model="regression",
                    PCA_m=i,
                    LDA_m=j,
                    l=l,
                )
                tableKFold[cont].append([accuracy, DCFnorm, minDCF])
                cont += 1

    print("PCA with k-fold")
    print(tabulate(tableKFold, headers=headers))
