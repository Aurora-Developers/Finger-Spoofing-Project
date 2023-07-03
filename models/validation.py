import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

tableTrain = []
tableTest = []
headers = [
    "Gaussian MVG PCA 8  kfold=20",
    "GMM Full Full, [0.1, 2, 0.01]",
    "SVM Polynomial PCA 7, [1, 0, 2 ,0]",
    "LR PCA8 LDA3 kfold=20 l=10-6",
    "QR PCA7 k_fold=20 l=10-1",
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


def calculate_model(
    args,
    test_points,
    model,
    prior_probability,
    test_labels,
    train_att=[],
    train_labels=[],
):
    model = model.lower()
    funct = lambda s: 1 if s > 0 else 0
    if model == "gaussian":
        multi_mu = args[0]
        cov = args[1]
        densities = []
        for i in np.unique(test_labels):
            densities.append(
                ML.logLikelihood(test_points, ML.vcol(multi_mu[i, :]), cov[i])
            )
        S = np.array(densities)
        logSJoint = S + np.log(prior_probability)
        logSMarginal = ML.vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)
        predictions = np.argmax(SPost, axis=0)
    elif model == "regression":
        w = args[0]
        b = args[1]
        S = np.dot(w.T, test_points) + b
        predictions = np.array(list(map(funct, S)))
    elif model == "quadratic":
        xt_2 = np.dot(test_points.T, test_points).diagonal().reshape((1, -1))
        test_points = np.vstack((xt_2, test_points))
        w = args[0]
        b = args[1]
        S = np.dot(w.T, test_points) + b
        predictions = np.array(list(map(funct, S)))
    elif model == "gmm":
        class_mu = args[0]
        class_c = args[1]
        class_w = args[2]
        densities = []
        for i in np.unique(test_labels):
            ll = np.array(ML.ll_gaussian(test_points, class_mu[i], class_c[i]))
            Sjoin = ll + np.log(class_w[i].reshape((-1, 1)))
            logdens = scipy.special.logsumexp(Sjoin, axis=0)
            densities.append(logdens)
        S = np.array(densities)
        predictions = np.argmax(S, axis=0)
    elif model == "svm":
        x = args[0]
        zi = 2 * train_label - 1
        S = x * zi
        S = np.dot(S, ML.polynomial_kernel(train_att, test_points, 2, 0, 0))
        predictions = np.where(S > 0, 1, 0)
    confusion_matrix = ML.ConfMat(predictions, test_labels)
    DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    (minDCF, _, _) = ML.minCostBayes(S, test_labels, pi, Cfn, Cfp)
    error = np.abs(test_labels - predictions)
    error = np.sum(error) / test_labels.shape[0]
    return predictions, (1 - error), DCFnorm, minDCF


if __name__ == "__main__":
    l_list = [0.01]
    minDCf7_list = []
    minDCffull_list = []
    minDCf8_list = []

    tot_iter = 10
    perc = 1

    pathTrain = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(pathTrain)
    pathTest = os.path.abspath("data/Test.txt")

    [full_test_att, full_test_labels] = load(pathTest)

    priorProb = ML.vcol(np.ones(2) * 0.5)
    tableTest.append([])
    tableTrain.append([])
    (train_att, train_label), (validation_att, validation_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )

    [_, Predictions, accuracy, DCFnorm, minDCF, mu, cov, P] = ML.k_fold(
        20, full_train_att, full_train_label, priorProb, model="mvg", final=1, PCA_m=8
    )
    tableTrain[0].append([accuracy, DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    PCA_8_test = np.dot(P.T, full_test_att)
    [Predictions, acc, DCFnorm, minDCF] = calculate_model(
        [mu, cov], PCA_8_test, "gaussian", priorProb, full_test_labels
    )
    tableTest[0].append([round(acc * 100, 2), DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    [SPost, Predictions, accuracy, mu, cov, w] = ML.Generative_models(
        train_att,
        train_label,
        validation_att,
        priorProb,
        validation_labels,
        "gmm",
        niter=2,
        alpha=0.1,
        psi=0.01,
        final=1,
    )
    confusion_matrix = ML.ConfMat(Predictions, validation_labels)
    DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    (minDCF, FPRlist, FNRlist) = ML.minCostBayes(SPost, validation_labels, pi, Cfn, Cfp)
    tableTrain[0].append([accuracy, DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    [Predictions, acc, DCFnorm, minDCF] = calculate_model(
        [mu, cov, w], full_test_att, "gmm", priorProb, full_test_labels
    )
    tableTest[0].append([round(acc * 100, 2), DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    [P, PCA_7_train_att] = ML.PCA(train_att, 7)
    PCA_7_validation_att = np.dot(P.T, validation_att)
    [SPost, Predictions, accuracy, x] = ML.svm(
        PCA_7_train_att,
        train_label,
        PCA_7_validation_att,
        validation_labels,
        constrain=1,
        dim=2,
        c=0,
        eps=0**2,
        model="polynomial",
        final=1,
    )
    confusion_matrix = ML.ConfMat(Predictions, validation_labels)
    DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    (minDCF, FPRlist, FNRlist) = ML.minCostBayes(SPost, validation_labels, pi, Cfn, Cfp)
    tableTrain[0].append([accuracy, DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1
    PCA_7_test = np.dot(P.T, full_test_att)
    [Predictions, acc, DCFnorm, minDCF] = calculate_model(
        [x],
        PCA_7_test,
        "svm",
        priorProb,
        full_test_labels,
        train_att=PCA_7_train_att,
        train_labels=train_label,
    )
    tableTest[0].append([round(acc * 100, 2), DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    [_, Predictions, accuracy, DCFnorm, minDCF, w, b, P, L] = ML.k_fold(
        20,
        full_train_att,
        full_train_label,
        priorProb,
        model="regression",
        PCA_m=8,
        LDA_m=3,
        l=10**-6,
        final=1,
    )
    tableTrain[0].append([accuracy, DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1
    LDA_test = np.dot(L.T, np.dot(P.T, full_test_att))
    [Predictions, acc, DCFnorm, minDCF] = calculate_model(
        [w, b], LDA_test, "regression", priorProb, full_test_labels
    )
    tableTest[0].append([round(acc * 100, 2), DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

    [_, Predictions, accuracy, DCFnorm, minDCF, w, b, PQ] = ML.k_fold(
        20,
        full_train_att,
        full_train_label,
        priorProb,
        model="regression",
        PCA_m=7,
        l=10**-1,
        final=1,
        quadratic=1,
    )
    tableTrain[0].append([accuracy, DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1
    PCA_test = np.dot(PQ.T, full_test_att)
    [Predictions, acc, DCFnorm, minDCF] = calculate_model(
        [w, b], PCA_test, "quadratic", priorProb, full_test_labels
    )
    tableTest[0].append([round(acc * 100, 2), DCFnorm, minDCF])
    print(f"{round(perc * 100 / tot_iter, 2)}%")
    perc += 1

print("Training metrics")
print(tabulate(tableTrain, headers=headers))
print()
print("Testing metrics")
print(tabulate(tableTest, headers=headers))
