import numpy as np
import pandas as pd
from MLandPattern import MLandPattern as ML
from tabulate import tabulate

table = []
headers = ["MVG", "Naive", "Tied Gaussian", "Tied Naive"]


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


def Generative_models(
    train_attributes, train_labels, test_attributes, prior_prob, test_labels, model
):
    if model.lower() == "mvg":
        [Probabilities, Prediction, accuracy] = ML.MVG_log_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
    elif model.lower() == "naive":
        [Probabilities, Prediction, accuracy] = ML.Naive_log_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
    elif model.lower() == "tied gaussian":
        [Probabilities, Prediction, accuracy] = ML.TiedGaussian(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
        accuracy = round(accuracy * 100, 2)
    elif model.lower() == "tied naive":
        [Probabilities, Prediction, accuracy] = ML.Tied_Naive_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
        accuracy = round(accuracy * 100, 2)
    return Probabilities, Prediction, accuracy


if __name__ == "__main__":
    [full_train_att, full_train_label] = load(
        "/Users/pablomunoz/Desktop/Polito 2023-1/MachineLearning/Project/data/Train.txt"
    )

    priorProb = ML.vcol(np.ones(2) * 0.5)

    (train_att, train_label), (test_att, test_labels) = split_db(
        full_train_att, full_train_label, 2 / 3
    )

    table.append(["Full"])

    [MVGprob, MVGpredic, accuracy] = ML.MVG_log_classifier(
        train_att, train_label, test_att, priorProb, test_labels
    )
    table[0].append(accuracy)

    [Naiveprob, Naivepredic, accuracy] = ML.Naive_log_classifier(
        train_att, train_label, test_att, priorProb, test_labels
    )
    table[0].append(accuracy)

    [MVGprob, MVGpredic, accuracy] = ML.TiedGaussian(
        train_att, train_label, test_att, priorProb, test_labels
    )
    table[0].append(round(accuracy * 100, 2))

    [Naiveprob, Naivepredic, accuracy] = ML.Tied_Naive_classifier(
        train_att, train_label, test_att, priorProb, test_labels
    )
    table[0].append(round(accuracy * 100, 2))

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)

        table.append([f"PCA {i}"])
        for model in headers:
            [MVGprob, MVGpredic, accuracy] = Generative_models(
                reduced_train, train_label, reduced_test, priorProb, test_labels, model
            )
            table[cont].append(accuracy)
        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            table.append([f"PCA {i} LDA {j}"])
            W, _ = ML.LDA1(reduced_train, train_label, j)
            LDA_train = np.dot(W.T, reduced_train)
            LDA_test = np.dot(W.T, reduced_test)
            for model in headers:
                [MVGprob, MVGpredic, accuracy] = Generative_models(
                    LDA_train, train_label, LDA_test, priorProb, test_labels, model
                )
                table[cont].append(accuracy)
            cont += 1
    headers = ["Dimensions"] + headers
    print(tabulate(table, headers=headers))
