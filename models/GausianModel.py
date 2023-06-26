import numpy as np
import pandas as pd
from MLandPattern import MLandPattern as ML
import scipy
import os
from tabulate import tabulate

tablePCA = []
tableKFold = []
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


# TODO Remove all the testing functions


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


def covariance_within_class(matrix_values, label):
    """
    Calculates the average covariance within all the classes in a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return: a matrix with the total average covariance within each class
    """
    class_labels = np.unique(label)
    within_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    n = matrix_values.size
    for i in class_labels:
        centered_matrix = ML.center_data(matrix_values[:, label == i])
        cov_matrix = ML.covariance(centered_matrix, 1)
        cov_matrix = np.multiply(cov_matrix, centered_matrix.size)
        within_cov = np.add(within_cov, cov_matrix)
    within_cov = np.divide(within_cov, n)
    return within_cov


def covariance_between_class(matrix_values, label):
    """
    Calculates the total covariance between all the classes in a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return: a matrix with the covariance between each class
    """
    class_labels = np.unique(label)
    between_cov = np.zeros((matrix_values.shape[0], matrix_values.shape[0]))
    N = matrix_values.size
    m_general = ML.mean_of_matrix_rows(matrix_values)
    for i in range(len(class_labels)):
        values = matrix_values[:, label == i]
        nc = values.size
        m_class = ML.mean_of_matrix_rows(values)
        norm_means = np.subtract(m_class, m_general)
        matr = np.multiply(nc, np.dot(norm_means, norm_means.T))
        between_cov = np.add(between_cov, matr)
    between_cov = np.divide(between_cov, N)
    return between_cov


def LDA1(matrix_values, label, m):
    """
    Calculates the Lineal Discriminant Analysis to perform dimension reduction
    :param matrix_values: matrix with the datapoints, with each row being a point
    :param label: vector with the label values associated with the dataset
    :param m: number of dimensions of the targeted sub-space
    :return: the LDA directions matrix (W), and the orthogonal sub-space of the directions (U)
    """
    class_labels = np.unique(label)
    Sw = covariance_within_class(matrix_values, label)
    Sb = covariance_between_class(matrix_values, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    return W, U


def k_fold(k, attributes, labels, previous_prob, model="mvg", PCA_m=0, LDA_m=0):
    """
    Applies a k-fold cross validation on the dataset, applying the specified model.
    :param: `k` Number of partitions to divide the dataset
    :param: `attributes` matrix containing the whole training attributes of the dataset
    :param: `labels` the label vector related to the attribute matrix
    :param: `previous_prob` the column vector related to the prior probability of the dataset
    :param: `model` (optional). Defines the model to be applied to the model:
        - `mvg`: Multivariate Gaussian Model
        - `Naive`: Naive Bayes Classifier
        - `Tied Gaussian`: Tied Multivariate Gaussian Model
        - `Tied naive`: Tied Naive Bayes Classifier
    :return final_acc: Accuracy of the validation set
    :return final_S: matrix associated with the probability array
    """
    section_size = int(attributes.shape[1] / k)
    cont = 0
    low = 0
    high = section_size
    final_acc = -1
    model = model.lower()
    for i in range(k):
        if not i:
            validation_att = attributes[:, low:high]
            validation_labels = labels[low:high]
            train_att = attributes[:, high:]
            train_labels = labels[high:]
            if PCA_m:
                P, train_att = ML.PCA(train_att, PCA_m)
                validation_att = np.dot(P.T, validation_att)
                if LDA_m:
                    W, _ = ML.LDA1(train_att, train_labels, LDA_m)
                    train_att = np.dot(W.T, train_att)
                    validation_att = np.dot(W.T, validation_att)
            [S, _, acc] = Generative_models(
                train_att,
                train_labels,
                validation_att,
                previous_prob,
                validation_labels,
                model,
            )
            final_acc = acc
            final_S = S
            continue
        low += section_size
        high += section_size
        if high > attributes.shape[1]:
            high = attributes.shape
        validation_att = attributes[:, low:high]
        validation_labels = labels[low:high]
        train_att = attributes[:, :low]
        train_labels = labels[:low]
        train_att = np.hstack((train_att, attributes[:, high:]))
        train_labels = np.hstack((train_labels, labels[high:]))
        if PCA_m:
            P, train_att = ML.PCA(train_att, PCA_m)
            validation_att = np.dot(P.T, validation_att)
            if LDA_m:
                W, _ = ML.LDA1(train_att, train_labels, LDA_m)
                train_att = np.dot(W.T, train_att)
                validation_att = np.dot(W.T, validation_att)
        [S, _, acc] = Generative_models(
            train_att,
            train_labels,
            validation_att,
            previous_prob,
            validation_labels,
            model,
        )
        final_acc += acc
        final_S += S
    final_acc /= k
    final_S /= k
    return final_acc, final_S


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    ### ------------- PCA WITH 2/3 SPLIT ---------------------- ####
    (train_att, train_label), (test_att, test_labels) = ML.split_db(
        full_train_att, full_train_label, 2 / 3
    )
    tablePCA.append(["Full"])

    for model in headers:
        [modelS, _, accuracy] = ML.Generative_models(
            train_att, train_label, test_att, priorProb, test_labels, model
        )
        tablePCA[0].append(accuracy)

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break
        P, reduced_train = ML.PCA(train_att, i)
        reduced_test = np.dot(P.T, test_att)

        tablePCA.append([f"PCA {i}"])
        for model in headers:
            [modelS, _, accuracy] = ML.Generative_models(
                reduced_train, train_label, reduced_test, priorProb, test_labels, model
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
            for model in headers:
                [modelS, _, accuracy] = ML.Generative_models(
                    LDA_train, train_label, LDA_test, priorProb, test_labels, model
                )
                tablePCA[cont].append(accuracy)
            cont += 1

    print("PCA with a 2/3 split")
    print(tabulate(tablePCA, headers=headers))

    ### ------------- k-fold with different PCA ---------------------- ####
    headers = headers[1:]
    tableKFold.append(["Full"])
    print(f"Size of dataset: {full_train_att.shape[1]}")
    k_fold_value = int(input("Value for k partitions: "))
    for model in headers:
        [accuracy, model] = ML.k_fold(
            k_fold_value, full_train_att, full_train_label, priorProb, model
        )
        tableKFold[0].append(accuracy)

    cont = 1
    for i in reversed(range(10)):
        if i < 2:
            break

        tableKFold.append([f"PCA {i}"])
        for model in headers:
            [accuracy, model] = ML.k_fold(
                k_fold_value,
                full_train_att,
                full_train_label,
                priorProb,
                model,
                PCA_m=i,
            )
            tableKFold[cont].append(accuracy)

        cont += 1
        for j in reversed(range(i)):
            if j < 2:
                break
            tableKFold.append([f"PCA {i} LDA {j}"])
            for model in headers:
                [accuracy, model] = ML.k_fold(
                    k_fold_value,
                    full_train_att,
                    full_train_label,
                    priorProb,
                    model,
                    PCA_m=i,
                    LDA_m=j,
                )
                tableKFold[cont].append(accuracy)
            cont += 1

    print("PCA with k-fold")
    print(tabulate(tableKFold, headers=headers))
