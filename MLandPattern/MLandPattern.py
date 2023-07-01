import numpy as np
import pandas as pd
import scipy
import math
from matplotlib import pyplot as plt


def loadCSV(pathname, class_label, attribute_names):
    """
    Extracts the attributes and class labels of an input
    csv file dataset
    All arguments must be of equal length.
    :param pathname: path to the data file
    :param class_label: list with class label names
    :param attribute_names: list with attribute names
    :return: two numpy arrays, one with the attributes and another
            with the class labels as numbers, ranging from [0, n]
    """
    # Read the CSV file
    df = pd.read_csv(pathname, header=None)

    # Extract the attributes from the dataframe
    attribute = np.array(df.iloc[:, 0 : len(attribute_names)])
    attribute = attribute.T

    # Re-assign the values of the class names to numeric values
    label_list = []
    for lab in df.loc[:, len(attribute_names)]:
        label_list.append(class_label.index(lab))
    label = np.array(label_list)
    return attribute, label


def split_db(D, L, ratio, seed=0):
    """
    Splits a dataset D into a training set and a validation set, based on the ratio
    :param D: matrix of attributes of the dataset
    :param L: vector of labels of the dataset
    :param ratio: ratio used to divide the dataset (e.g. 2 / 3)
    :param seed: seed for the random number generator of numpy (default 0)
    :return (DTR, LTR), (DTE, LTE): (DTR, LTR) attributes and labels releated to the training sub-set. (DTE, LTE) attributes and labels releated to the testing sub-set.

    """
    nTrain = int(D.shape[1] * ratio)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def load(pathname, vizualization=0):
    """
    Loads simple csv, assuming first n-1 columns as attributes, and n col as class labels
    :param pathname: path to the data file
    :param vizualization: flag to determine if print on console dataframe head (default false)
    :return: attributes, labels. attrributes contains a numpy matrix with the attributes of the dataset. labels contains a numpy matrix
            with the class labels as numbers, ranging from [0, n]
    """
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def vcol(vector):
    """
    Reshape a vector row vector into a column vector
    :param vector: a numpy row vector
    :return: the vector reshaped as a column vector
    """
    column_vector = vector.reshape((vector.size, 1))
    return column_vector


def vrow(vector):
    """
    Reshape a vector column vector into a row vector
    :param vector: a numpy column vector
    :return: the vector reshaped as a row vector
    """
    row_vector = vector.reshape((1, vector.size))
    return row_vector


def mean_of_matrix_rows(matrix):
    """
    Calculates the mean of the rows of a matrix
    :param matrix: a matrix of numpy arrays
    :return: a numpy column vector with the mean of each row
    """
    mu = matrix.mean(1)
    mu_col = vcol(mu)
    return mu_col


def center_data(matrix):
    """
    Normalizes the data on the dataset by subtracting the mean
    to each element.
    :param matrix: a matrix of numpy arrays
    :return: a matrix of the input elements minus the mean for
    each row
    """
    mean = mean_of_matrix_rows(matrix)
    centered_data = matrix - mean
    return centered_data


def covariance(matrix, centered=0):
    """
    Calculates the Sample Covariance Matrix of a centered-matrix
    :param matrix: Matrix of data points
    :param centered: Flag to determine if matrix data is centered (default is False)
    :return: The data covariance matrix
    """
    if not centered:
        matrix = center_data(matrix)
    n = matrix.shape[1]
    cov = np.dot(matrix, matrix.T)
    cov = np.multiply(cov, 1 / n)
    return cov


def eigen(matrix):
    """
    Calculates the eigen value and vectors for a matrix
    :param matrix: Matrix of data points
    :return: eigen values, eigen vectors
    """
    if matrix.shape[0] == matrix.shape[1]:
        s, U = np.linalg.eigh(matrix)
        return s, U
    else:
        s, U = np.linalg.eig(matrix)
        return s, U


def PCA(attribute_matrix, m):
    """
    Calculates the PCA dimension reduction of a matrix to a m-dimension sub-space
    :param attribute_matrix: matrix with the datapoints, with each row being a point
    `param m` number of dimensions of the targeted sub-space
    :return: The matrix P defined to do the PCA approximation
    :return: The dataset after the dimensionality reduction
    """
    DC = center_data(attribute_matrix)
    C = covariance(DC, 1)
    s, U = eigen(C)
    P = U[:, ::-1][:, 0:m]
    return P, np.dot(P.T, attribute_matrix)


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
        centered_matrix = center_data(matrix_values[:, label == i])
        cov_matrix = covariance(centered_matrix, 1)
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
    m_general = mean_of_matrix_rows(matrix_values)
    for i in range(len(class_labels)):
        values = matrix_values[:, label == i]
        nc = values.size
        m_class = mean_of_matrix_rows(values)
        norm_means = np.subtract(m_class, m_general)
        matr = np.multiply(nc, np.dot(norm_means, norm_means.T))
        between_cov = np.add(between_cov, matr)
    between_cov = np.divide(between_cov, N)
    return between_cov


def between_within_covariance(matrix_values, label):
    """
    Calculates both the average within covariance, and the between covariance of all classes on a dataset
    :param matrix_values: matrix with the values associated to the parameters of the dataset
    :param label: vector with the label values associated with the dataset
    :return:a matrix with the total average covariance within each class, and the covariance between each class
    """
    Sw = covariance_within_class(matrix_values, label)
    Sb = covariance_between_class(matrix_values, label)
    return Sw, Sb


def LDA1(matrix_values, label, m):
    """
    Calculates the Lineal Discriminant Analysis to perform dimension reduction
    :param matrix_values: matrix with the datapoints, with each row being a point
    :param label: vector with the label values associated with the dataset
    :param m: number of dimensions of the targeted sub-space
    :return: the LDA directions matrix (W), and the orthogonal sub-space of the directions (U)
    """
    class_labels = np.unique(label)
    [Sw, Sb] = between_within_covariance(matrix_values, label)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]
    return W, U


#  General method to graph a class-related data into a 2d scatter plot
def graphic_scatter_2d(matrix, labels, names, x_axis="Axis 1", y_axis="Axis 2"):
    for i in range(len(names)):
        plt.scatter(matrix[0][labels == i], matrix[1][labels == i], label=names[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()


def logpdf_GAU_ND(x, mu, C):
    """
    Calculates the Logarithmic MultiVariate Gaussian Density for a set of vector values
    :param x: matrix of the datapoints of a dataset, with a size (n x m)
    :param mu: row vector with the mean associated to each dimension
    :param C: Covariance matrix
    :return: a matrix with the Gaussian Density associated with each point of X, over each dimension
    """
    M = C.shape[1]
    inv_C = np.linalg.inv(C)
    # print(inv_C.shape)
    [_, log_C] = np.linalg.slogdet(C)

    # print(log_C)
    log_2pi = -M * math.log(2 * math.pi)
    x_norm = x - mu
    inter_value = np.dot(x_norm.T, inv_C)
    dot_mul = np.dot(inter_value, x_norm)
    dot_mul = np.diag(dot_mul)

    y = (log_2pi - log_C - dot_mul) / 2
    return y


def logLikelihood(X, mu, c, tot=0):
    """
    Calculates the Logarithmic Maximum Likelihood estimator
    :param X: matrix of the datapoints of a dataset, with a size (n x m)
    :param mu: row vector with the mean associated to each dimension
    :param c: Covariance matrix
    :param tot: flag to define if it returns value per datapoint, or total sum of logLikelihood (default is False)
    :return: the logarithm of the likelihood of the datapoints, and the associated gaussian density
    """
    M = c.shape[1]
    logN = logpdf_GAU_ND(X, mu, c)
    if tot:
        return logN.sum()
    else:
        return logN


def multiclass_covariance(matrix, labels):
    """
    Calculates the Covariance for each class in  dataset
    :param matrix: matrix of the datapoints of a dataset
    :param labels: row vector with the labels associated with each row of data points
    :return: A np matrix with size (# of classes, n, n) related with the covariance associated with each class, in each dimension
    """
    class_labels = np.unique(labels)
    within_cov = np.zeros((class_labels.size, matrix.shape[0], matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        centered_matrix = center_data(matrix[:, labels == i])
        within_cov[i, :, :] = covariance(centered_matrix)
    return within_cov


def multiclass_mean(matrix, labels):
    """
    Calculates the mean for each class in  dataset
    :param matrix: matrix of the datapoints of a dataset
    :param labels: row vector with the labels associated with each row of data points
    :return: A np matrix with size (# of classes, n) related with the mean associated with each class, in each dimension
    """
    class_labels = np.unique(labels)
    multi_mu = np.zeros((class_labels.size, matrix.shape[0]))
    n = matrix.size
    for i in class_labels:
        mu = mean_of_matrix_rows(matrix[:, labels == i])
        multi_mu[i, :] = mu[:, 0]
    return multi_mu


def MVG_classifier(train_data, train_labels, test_data, test_label, prior_probability):
    """
    Calculates the model of the MultiVariate Gaussian classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    # print(cov[0])
    multi_mu = multiclass_mean(train_data, train_labels)
    # print(multi_mu[0])
    densities = []
    for i in range(class_labels.size):
        densities.append(np.exp(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i])))
    S = np.array(densities)
    SJoint = S * prior_probability
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        # print(f'Accuracy: {acc}%')
        # print(f'Error: {(100 - acc)}%')

    return S, predictions, acc


def MVG_log_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the MultiVariate Gaussian classifier on the logarithm dimension for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    # print(cov[0])
    multi_mu = multiclass_mean(train_data, train_labels)
    # print(multi_mu[0])
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i]))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        # print(f'Accuracy: {acc}%')
        # print(f'Error: {(100 - acc)}%')

    return S, predictions, acc


def Naive_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the Naive classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    identity = np.eye(cov.shape[1])
    # print(identity)
    cov = cov * identity
    multi_mu = multiclass_mean(train_data, train_labels)
    # print(multi_mu[0])
    densities = []
    for i in range(class_labels.size):
        densities.append(np.exp(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i])))
    S = np.array(densities)
    SJoint = S * prior_probability
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        # print(f'Accuracy: {acc}%')
        # print(f'Error: {(100 - acc)}%')

    return S, predictions, acc


def Naive_log_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the Naive classifier on the logarithm realm for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = multiclass_covariance(train_data, train_labels)
    identity = np.eye(cov.shape[1])
    # print(identity)
    cov = cov * identity
    multi_mu = multiclass_mean(train_data, train_labels)
    # print(multi_mu[0])
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov[i]))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        acc = round(acc * 100, 2)
        # print(f'Accuracy: {acc}%')
        # print(f'Error: {(100 - acc)}%')

    return S, predictions, acc


def TiedGaussian(train_data, train_labels, test_data, prior_probability, test_label=[]):
    """
    Calculates the model of the Tied Gaussian classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    with_cov = covariance_within_class(train_data, train_labels)
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), with_cov))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        # print(f"Accuracy: {acc}")
        # print(f"Error: {1 - acc}")

    return S, predictions, acc


def Tied_Naive_classifier(
    train_data, train_labels, test_data, prior_probability, test_label=[]
):
    """
    Calculates the model of the Tied Naive classifier for a set of data, and applyes it to a test dataset
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    class_labels = np.unique(train_labels)
    cov = covariance_within_class(train_data, train_labels)
    identity = np.eye(cov.shape[1])
    cov = cov * identity
    multi_mu = multiclass_mean(train_data, train_labels)
    densities = []
    for i in range(class_labels.size):
        densities.append(logLikelihood(test_data, vcol(multi_mu[i, :]), cov))
    S = np.array(densities)
    logSJoint = S + np.log(prior_probability)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)
    predictions = np.argmax(SPost, axis=0)

    if len(test_label) != 0:
        acc = 0
        for i in range(len(test_label)):
            if predictions[i] == test_label[i]:
                acc += 1
        acc /= len(test_label)
        # print(f"Accuracy: {acc*100}%")
        # print(f"Error: {(1 - acc)*100}%")

    return S, predictions, acc


def Generative_models(
    train_attributes, train_labels, test_attributes, prior_prob, test_labels, model
):
    """
    Calculates the desired generative model
    :param train_date: matrix of the datapoints of a dataset used to train the model
    :param train_labels: row vector with the labels associated with each row of the training dataset
    :param test_data: matrix of the datapoints of a dataset used to test the model
    :param test_labels: row vector with the labels associated with each row of the test dataset
    :param prior_probability: col vector associated with the prior probability for each dimension
    :param: `model`defines which model, based on the following criterias:
        - `mvg`: Multivariate Gaussian Model
        - `Naive`: Naive Bayes Classifier
        - `Tied Gaussian`: Tied Multivariate Gaussian Model
        - `Tied naive`: Tied Naive Bayes Classifier
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    if model.lower() == "mvg":
        [Probabilities, Prediction, accuracy] = MVG_log_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
    elif model.lower() == "naive":
        [Probabilities, Prediction, accuracy] = Naive_log_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
    elif model.lower() == "tied gaussian":
        [Probabilities, Prediction, accuracy] = TiedGaussian(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
        accuracy = round(accuracy * 100, 2)
    elif model.lower() == "tied naive":
        [Probabilities, Prediction, accuracy] = Tied_Naive_classifier(
            train_attributes, train_labels, test_attributes, prior_prob, test_labels
        )
        accuracy = round(accuracy * 100, 2)
    return Probabilities, Prediction, accuracy


def k_fold(
    k, attributes, labels, previous_prob, model="mvg", PCA_m=0, LDA_m=0, l=0.001
):
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
        - `Regression` : Binomial Regression
    :param: `PCA_m` (optional) a number of dimensions to reduce using PCA method
    :param: `LDA_m` (optional) a number of dimensions to reduce using LDA mehtod
    :param: `l` (optional) hyperparameter to use when the method is linera regression, default value set to 0.001
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
                P, train_att = PCA(train_att, PCA_m)
                validation_att = np.dot(P.T, validation_att)
                if LDA_m:
                    W, _ = LDA1(train_att, train_labels, LDA_m)
                    train_att = np.dot(W.T, train_att)
                    validation_att = np.dot(W.T, validation_att)
            if model == "regression":
                [S, prediction, acc] = binaryRegression(
                    train_att, train_labels, l, validation_att, validation_labels
                )
            else:
                [S, prediction, acc] = Generative_models(
                    train_att,
                    train_labels,
                    validation_att,
                    previous_prob,
                    validation_labels,
                    model,
                )
            final_acc = acc
            final_S = S
            final_prediction = prediction
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
            P, train_att = PCA(train_att, PCA_m)
            validation_att = np.dot(P.T, validation_att)
            if LDA_m:
                W, _ = LDA1(train_att, train_labels, LDA_m)
                train_att = np.dot(W.T, train_att)
                validation_att = np.dot(W.T, validation_att)
        if model == "regression":
            [S, prediction, acc] = binaryRegression(
                train_att, train_labels, l, validation_att, validation_labels
            )
        else:
            [S, prediction, acc] = Generative_models(
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
    return final_S, prediction, final_acc


def logreg_obj(v, DTR, LTR, l):
    """
    Method to calculate the error of a function based on the data points
    :param v: Vector of the values to evaluate [w, b]
    :param DTR: Matrix with all the train attributes
    :param LTR: Matrix with all the train labels
    :param l: Hyperparameter l to apply to the function
    :return: retFunc the value of evaluating the function on the input parameter v
    """
    n = DTR.shape[1]
    w, b = v[0:-1], v[-1]
    log_sum = 0
    zi = LTR * 2 - 1
    inter_sol = -zi * (np.dot(w.T, DTR) + b)
    log_sum = np.sum(np.logaddexp(0, inter_sol))
    retFunc = l / 2 * np.power(np.linalg.norm(w), 2) + 1 / n * log_sum
    return retFunc


def binaryRegression(train_attributes, train_labels, l, test_attributes, test_labels):
    """
    Method to calculate the error of a function based on the data points
    :param train_attributes: Matrix with all the train attributes
    :param train_labels: Matrix with all the train labels
    :param l: Hyperparameter l to apply to the function
    :param test_attributes: Matrix with all the train attributes
    :param test_labels: Matrix with all the train labels
    :return S: matrix associated with the probability array
    :return predictions: Vector associated with the prediction of the class for each test data point
    :return acc: Accuracy of the validation set
    """
    x0 = np.zeros(train_attributes.shape[0] + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        logreg_obj, x0, approx_grad=True, args=(train_attributes, train_labels, l)
    )
    w, b = x[0:-1], x[-1]
    S = np.dot(w.T, test_attributes) + b
    funct = lambda s: 1 if s > 0 else 0
    predictions = np.array(list(map(funct, S)))
    acc = 0
    for i in range(test_labels.shape[0]):
        if predictions[i] == test_labels[i]:
            acc += 1
    acc /= test_labels.size
    acc = round(acc * 100, 2)

    return w, S, acc


def polynomial_kernel(xi, xj, d, C, eps):
    interm = np.dot(xi.T, xj)
    interm += C
    G = np.power(interm, d) + eps
    return G


def radial_kernel(xi, xj, gamma, eps):
    diff = xi[:, :, np.newaxis] - xj[:, np.newaxis, :]
    G = -gamma * np.square(np.linalg.norm(diff, axis=0))
    G = np.add(np.exp(G), eps)
    return G


def dual_svm(
    alpha,
    training_att,
    training_labels,
    K=1,
    d=0,
    c=0,
    eps=0,
    gamma=0,
    funct="polynomial",
):
    kern = funct.lower()
    one = np.ones(training_att.shape[1])
    zi = 2 * training_labels - 1
    if kern == "polynomial":
        G = polynomial_kernel(training_att, training_att, d, c, eps)
    elif kern == "radial":
        G = radial_kernel(training_att, training_att, gamma, eps=eps)
    else:
        D = np.vstack((training_att, one * K))
        G = np.dot(D.T, D)
    z = np.outer(zi, zi)
    H = np.multiply(z, G)
    retFun = np.dot(alpha.T, H)
    retFun = np.dot(retFun, alpha) / 2
    retFun = retFun - np.dot(alpha.T, one)
    retGrad = np.dot(H, alpha)
    retGrad -= one
    return (retFun, retGrad)


def svm(
    training_att,
    training_labels,
    test_att,
    test_labels,
    constrain,
    model="",
    dim=2,
    c=1,
    K=1,
    gamma=1,
    eps=0,
):
    """
    Apply the Support Vector Machine model, using either one of the models described to approximate the soluction.
    :param train_att: Matrix with all the train attributes
    :param train_labels: Matrix with all the train labels
    :param test_att: Matrix with all the train attributes
    :param test_labels: Matrix with all the train labels
    :param constrain: Constrain of maximum value of the alpha vector
    :param model: (optional) define the applied kernel model:
    - `polynomial`: polynomial kernel of degree d
    - `Radial`: Radial Basis Function kernel
    - Default: leave empty, and the dual SVM method is applied
    :param dim: (optional) hyperparameter for polynomial method. `Default 2`
    :param c: (optional) hyperparameter for polynomial method. `Default 1`
    :param K: (optional) hyperparameter for dual SVM method. `Default 1`
    :param gamma: (optional) hyperparameter for radial method. `Default 1`
    :param eps: (optional) hyperparameter for kernel methods. `Default 0`
    """
    alp = np.ones(training_att.shape[1])
    constrain = np.array([(0, constrain)] * training_att.shape[1])
    [x, f, d] = scipy.optimize.fmin_l_bfgs_b(
        dual_svm,
        alp,
        args=(training_att, training_labels, K, dim, c, eps, gamma, model),
        bounds=constrain,
    )
    print(d["nit"])
    zi = 2 * training_labels - 1
    kern = model.lower()
    if kern == "polynomial":
        S = x * zi
        S = np.dot(S, polynomial_kernel(training_att, test_att, dim, c, eps))
    elif kern == "radial":
        S = x * zi
        S = np.dot(S, radial_kernel(training_att, test_att, gamma, eps))
    else:
        D = np.ones(training_att.shape[1]) * K
        D = np.vstack((training_att, D))
        w = x * zi
        w = w * D
        w = np.sum(w, axis=1)
        x_val = np.vstack((test_att, np.ones(test_att.shape[1]) * K))
        S = np.dot(w.T, x_val)
    # funct = lambda s: 1 if s > 0 else 0
    predictions = np.where(S > 0, 1, 0)
    error = np.abs(predictions - test_labels)
    error = np.sum(error)
    error /= test_labels.size

    return S, predictions, 1 - error


def calculate_model(S, test_points, model, prior_probability, test_labels=[]):
    model = model.lower()
    funct = lambda s: 1 if s > 0 else 0
    if model == "Generative":
        logSJoint = S + np.log(prior_probability)
        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        SPost = np.exp(logSPost)
        predictions = np.argmax(SPost, axis=0)
    elif model == "regression":
        predictions = np.array(list(map(funct, S)))
    else:
        predictions = np.array(list(map(funct, S)))
    if len(test_labels) != 0:
        error = np.abs(test_labels - predictions)
        error = np.sum(error)
    return predictions, (1 - error)


def ConfMat(predicted, actual):
    labels = np.unique(np.concatenate((actual, predicted)))

    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, predicted_label in zip(actual, predicted):
        true_index = np.where(labels == true_label)[0]
        predicted_index = np.where(labels == predicted_label)[0]
        matrix[predicted_index, true_index] += 1

    return matrix


def OptimalBayes(llr, labels, pi, Cfn, Cfp):
    log_odds = llr
    threshold = -np.log((pi * Cfn) / ((1 - pi) * Cfp))
    decisions = np.where(log_odds > threshold, 1, 0)

    tp = np.sum(np.logical_and(decisions == 1, labels == 1))
    fp = np.sum(np.logical_and(decisions == 1, labels == 0))
    tn = np.sum(np.logical_and(decisions == 0, labels == 0))
    fn = np.sum(np.logical_and(decisions == 0, labels == 1))

    confusion_matrix = np.array([[tn, fn], [fp, tp]])

    return confusion_matrix


def Bayes_risk(confusion_matrix, pi, Cfn, Cfp):
    M01 = confusion_matrix[0][1]
    M11 = confusion_matrix[1][1]
    M10 = confusion_matrix[1][0]
    M00 = confusion_matrix[0][0]

    FNR = M01 / (M01 + M11)
    FPR = M10 / (M00 + M10)

    DCF = pi * Cfn * FNR + (1 - pi) * Cfp * FPR

    B = min(pi * Cfn, (1 - pi) * Cfp)

    DCFnorm = DCF / B

    return DCF, round(DCFnorm, 2)


def minCostBayes(llr, labels, pi, Cfn, Cfp):
    if llr.ndim > 1:
        llr = (Cfn * llr[0, :]) / (Cfp * llr[1, :])
    sortedLLR = np.sort(llr)
    # sortedLLR = pi * sortedLLR[0, :] / ((1 - pi) * sortedLLR[1, :])
    t = np.array([-np.inf, np.inf])
    t = np.append(t, sortedLLR)
    t = np.sort(t)
    DCFnorm = []
    FNRlist = []
    FPRlist = []
    for i in t:
        threshold = i
        funct = lambda s: 1 if s > i else 0
        decisions = np.array(list(map(funct, llr)))
        # decisions = np.where(llr > threshold, 1, 0)

        tp = np.sum(np.logical_and(decisions == 1, labels == 1))
        fp = np.sum(np.logical_and(decisions == 1, labels == 0))
        tn = np.sum(np.logical_and(decisions == 0, labels == 0))
        fn = np.sum(np.logical_and(decisions == 0, labels == 1))

        confusion_matrix = np.array([[tn, fn], [fp, tp]])
        M01 = confusion_matrix[0][1]
        M11 = confusion_matrix[1][1]
        M10 = confusion_matrix[1][0]
        M00 = confusion_matrix[0][0]

        FNR = M01 / (M01 + M11)
        FPR = M10 / (M00 + M10)

        [DCF, DCFnormal] = Bayes_risk(confusion_matrix, pi, Cfn, Cfp)

        FNRlist = np.append(FNRlist, FNR)
        FPRlist = np.append(FPRlist, FPR)
        DCFnorm = np.append(DCFnorm, DCFnormal)
    minDCF = min(DCFnorm)

    return minDCF, FPRlist, FNRlist
