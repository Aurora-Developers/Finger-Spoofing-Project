import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class_label = ["0", "1"]
attribute_names = []
alpha_val = 0.3

for i in range(10):
    attribute_names.append(str(i))


def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    # print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


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


[attributes, labels] = load(
    "/Users/pablomunoz/Desktop/Polito 2023-1/MachineLearning/Project/data/Test.txt"
)
print(f"Attributes size: {attributes.size}")
print(f"Attributes shape: {attributes.shape}")
print(f"Labels size: {labels.size}")
print(f"Labels shape: {labels.shape}")


def graficar():
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

    dimension = input("Input: \n- Dimension to evalueate (1 to 10).\n- 0 to exit. \n")
    dimension = str(int(dimension) - 1)
    while dimension != "-1":
        xv = values_histogram[dimension]
        xk = dimension
        cont = 1
        plt.suptitle(f"Analyzing: dim {dimension}", fontsize=16)
        for yk, yv in values_histogram.items():
            if xk == yk:
                plt.subplot(5, 2, cont)
                histogram_1n(xv[0], xv[1], x_axis=xk)
                cont += 1
            else:
                plt.subplot(5, 2, cont)
                scatter_2d([xv[0], yv[0]], [xv[1], yv[1]], x_axis=xk, y_axis=yk)
                cont += 1
        plt.show()
        dimension = input(
            "Input: \n- Dimension to evalueate (1 to 10).\n- 11 to show all 10 dimensions. \n - 0 to exit: "
        )
        dimension = str(int(dimension) - 1)


def mcol(matrix, vector):
    column_vector = vector.reshape((matrix.shape[0], 1))
    return column_vector


def mrow(matrix, vector):
    row_vector = vector.reshape((1, matrix.shape[0]))
    return row_vector


graficar()
