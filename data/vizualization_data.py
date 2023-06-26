import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

class_label = ["0", "1"]
attribute_names = []
alpha_val = 0.5


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


jhb


def histogram_1n(setosa, versicolor, x_axis="", y_axis=""):
    plt.hist(setosa, color="blue", alpha=alpha_val, label=class_label[0], density=True)
    plt.hist(
        versicolor, color="orange", alpha=alpha_val, label=class_label[1], density=True
    )
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def scatter_2d(setosa, versicolor, x_axis="", y_axis=""):
    plt.scatter(setosa[0], setosa[1], c="blue", s=1.5)
    plt.scatter(versicolor[0], versicolor[1], c="orange", s=1.5)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)


def graficar(attributes):
    attribute_names = []
    for i in range(attributes.shape[0]):
        attribute_names.append(str(i))
    values_histogram = {}

    for i in range(len(attribute_names)):
        values_histogram[attribute_names[i]] = [
            attributes[i, labels == 0],
            attributes[i, labels == 1],
        ]

    for a in attribute_names:
        histogram_1n(
            values_histogram[a][0],
            values_histogram[a][1],
            x_axis=a,
        )

    size1 = round(attributes.shape[0] / 2 + 0.5)
    if attributes.shape[0] > 8:
        dimension = input(
            f"Input: \n- Dimension to evalueate (1 to {attributes.shape[0]}).\n- 0 to exit. \n"
        )
        dimension = str(int(dimension) - 1)
        while dimension != "-1":
            xv = values_histogram[dimension]
            xk = dimension
            cont = 1
            plt.suptitle(f"Analyzing: dim {dimension}", fontsize=16)
            for yk, yv in values_histogram.items():
                if xk == yk:
                    plt.subplot(size1, 2, cont)
                    histogram_1n(xv[0], xv[1], x_axis=xk)
                    cont += 1
                else:
                    plt.subplot(size1, 2, cont)
                    scatter_2d([xv[0], yv[0]], [xv[1], yv[1]], x_axis=xk, y_axis=yk)
                    cont += 1
            plt.show()
            dimension = input(
                f"Input: \n- Dimension to evalueate (1 to {attributes.shape[0]}). \n - 0 to exit: "
            )
            dimension = str(int(dimension) - 1)
    else:
        cont = 1
        for xk, xv in values_histogram.items():
            for yk, yv in values_histogram.items():
                if xk == yk:
                    plt.subplot(attributes.shape[0], attributes.shape[0], cont)
                    histogram_1n(xv[0], xv[1], x_axis=xk)
                    cont += 1
                else:
                    plt.subplot(attributes.shape[0], attributes.shape[0], cont)
                    scatter_2d([xv[0], yv[0]], [xv[1], yv[1]], x_axis=xk, y_axis=yk)
                    cont += 1
        plt.show()


def independent_graph(attributes):
    attribute_names = []
    for i in range(attributes.shape[0]):
        attribute_names.append(str(i))
    values_histogram = {}

    for i in range(len(attribute_names)):
        values_histogram[attribute_names[i]] = [
            attributes[i, labels == 0],
            attributes[i, labels == 1],
        ]

    for a in attribute_names:
        histogram_1n(
            values_histogram[a][0],
            values_histogram[a][1],
            x_axis=a,
        )

    # for i in values_histogram.items():
    #     xv = values_histogram[dimension]
    #     xk = dimension
    #     cont = 1
    #     plt.suptitle(f"Analyzing: dim {dimension}", fontsize=16)
    cont = 1
    for yk, yv in values_histogram.items():
        plt.subplot(5, 2, cont)
        histogram_1n(yv[0], yv[1], x_axis=yk)
        cont += 1
    plt.show()


def mcol(matrix, vector):
    column_vector = vector.reshape((matrix.shape[0], 1))
    return column_vector


def mrow(matrix, vector):
    row_vector = vector.reshape((1, matrix.shape[0]))
    return row_vector


def PCA(attribute_matrix, m):
    """
    Calculates the PCA dimension reduction of a matrix to a m-dimension sub-space
    :param attribute_matrix: matrix with the datapoints, with each row being a point
    :param m: number of dimensions of the targeted sub-space
    :return: The dataset after the dimensionality reduction
    """
    DC = ML.center_data(attribute_matrix)
    C = ML.covariance(attribute_matrix)
    s, U = ML.eigen(C)
    P = U[:, ::-1][:, 0:m]
    return np.dot(P.T, attribute_matrix)


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [attributes, labels] = load(path)
    print(f"Attribute dimensions: {attributes.shape[0]}")
    print(f"Points on the dataset: {attributes.shape[1]}")
    print(f"Possible classes: {class_label[0]}, {class_label[1]}")
    configure = input("Analyze whole dataset (1), or a PCA reduction (2) exit(0): ")
    while configure != "0":
        if configure == "1":
            independent_graph(attributes)
            graficar(attributes)
        else:
            m = int(input("Number of dimensions: "))
            _, copy_attributes = ML.PCA(attributes, m)
            independent_graph(copy_attributes)
            graficar(copy_attributes)
        configure = input("Analyze whole dataset (1), or a PCA reduction (2) exit(0): ")
