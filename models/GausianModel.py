import numpy as np
import pandas as pd
from MLandPattern import MLandPattern as ML


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
    [full_train_att, full_train_label] = load(
        "/Users/pablomunoz/Desktop/Polito 2023-1/MachineLearning/Project/data/Train.txt"
    )

    priorProb = ML.vcol(np.ones(2) * 0.5)

    (train_att, train_label), (test_att, test_labels) = split_db(
        full_train_att, full_train_label, 2 / 3
    )

    print("Prediction with full dimensions: ")
    print("MVG: ")

    [MVGprob, MVGpredic] = ML.MVG_log_classifier(
        train_att, train_label, test_att, priorProb, test_labels
    )
    print("Naive Bayes")
    [Naiveprob, Naivepredic] = ML.Naive_log_classifier(
        train_att, train_label, test_att, priorProb, test_labels
    )
    print()

    for i in reversed(range(10)):
        if i < 2:
            break
        reduced_train = np.dot(ML.PCA(train_att, i).T, train_att)
        reduced_test = np.dot(ML.PCA(test_att, i).T, test_att)
        print(f"Prediction with {i} dimensions: ")
        print("MVG: ")
        [MVGprob, MVGpredic] = ML.MVG_log_classifier(
            reduced_train, train_label, reduced_test, priorProb, test_labels
        )
        print("Naive Bayes")
        [Naiveprob, Naivepredic] = ML.Naive_log_classifier(
            reduced_train, train_label, reduced_test, priorProb, test_labels
        )
        print()
