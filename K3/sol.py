# ###################################
# Group ID : 713
# Members : Anders Lauridsen,
#           Isabella Quillo,
#           Jacob Mørk,
#           Jakob Olsen
# Date : 14/9/2022
# Lecture: 3, Parametric and nonparametric methods
# Dependencies: Numpy and Scipy
# Python version: 3.10
# Functionality: Classifies points into classes using probability theory
# ###################################
import numpy as np
from scipy.stats import multivariate_normal


# %% Load the data
trn_x = np.loadtxt("data/trn_x.txt")
trn_y = np.loadtxt("data/trn_y.txt")
trn_x_class = np.loadtxt("data/trn_x_class.txt")
trn_y_class = np.loadtxt("data/trn_y_class.txt")

tst_x = np.loadtxt("data/tst_x.txt")
tst_y = np.loadtxt("data/tst_y.txt")
tst_y_126 = np.loadtxt("data/tst_y_126.txt")
tst_xy = np.loadtxt("data/tst_xy.txt")
tst_xy_126 = np.loadtxt("data/tst_xy_126.txt")

tst_x_class = np.loadtxt("data/tst_x_class.txt")
tst_y_class = np.loadtxt("data/tst_y_class.txt")
tst_y_126_class = np.loadtxt("data/tst_y_126_class.txt")
tst_xy_class = np.loadtxt("data/tst_xy_class.txt")
tst_xy_126_class = np.loadtxt("data/tst_xy_126_class.txt")

# %% Task a)
"""
We compute posterior probability and assign the class based on the largest probability.
"""


def pick_class(point, mu_x, cov_x, mu_y, cov_y, p_cx, p_cy):
    """
    Computes the post. prob. and returns class number
    :param point: point to evaluate
    :param mu_x: mean of class x
    :param cov_x: cov of class x
    :param mu_y: mean of class y
    :param cov_y: cov of class y
    :param p_cx: prior prob. of class x
    :param p_cy: prior prob. of class y
    :return:
    """
    p_x = multivariate_normal.pdf(point, mu_x, cov_x) * p_cx
    p_y = multivariate_normal.pdf(point, mu_y, cov_y) * p_cy

    if p_x > p_y:
        return 1
    elif p_x < p_y:
        return 2
    else:
        choice = np.random.randint(0, 2)
        return choice


# Compute priors
N = len(trn_x_class) + len(trn_y_class)
p_cx = len(trn_x_class) / N
p_cy = len(trn_y_class) / N

# Compute means and covariances
m_x = np.mean(trn_x, 0)
m_y = np.mean(trn_y, 0)
cov_x = np.cov(trn_x.T)
cov_y = np.cov(trn_y.T)

# Label the data
labels = []
for point in tst_xy:
    result = pick_class(point, m_x, cov_x, m_y, cov_y, p_cx, p_cy)
    labels.append(result)

# Calculate the accuracy
num_correct = 0
num_correct_x = 0
num_correct_y = 0
for guess, true in zip(labels, tst_xy_class):
    if guess == true:
        num_correct += 1
        if true == 1:
            num_correct_x += 1
        else:
            num_correct_y += 1

# results
acc = num_correct / len(tst_xy_class)
acc_x = num_correct_x / len(tst_x_class)
acc_y = num_correct_y / len(tst_y_class)

print(f"The total accuracy of the classification in task a) is: {100*acc:.2f}%")
print(f"The accuracy when classifying point as x in task a) is: {100*acc_x:.2f}%")
print(f"The accuracy when classifying point as y in task a) is: {100*acc_y:.2f}%")
