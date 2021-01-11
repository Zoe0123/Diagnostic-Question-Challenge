from utils import *

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# size of students and questions
N, D = 542, 1774


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]

        # prob = p(c_uq|theta_u, beta_q)
        x = theta[u] - beta[q]
        prob = sigmoid(x)

        c = data["is_correct"][i]
        log_lklihood += c * np.log(prob) + (1 - c) * np.log((1 - prob))
    return -log_lklihood


def update_theta_beta(train_matrix, zero_train_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # create matrix representation for calculation convenience
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)

    nan_mask = np.isnan(train_matrix)

    # update theta
    x = theta_mat - beta_mat
    # probabilities p(c_uq|theta_u, beta_q) for each u and q
    prob = sigmoid(x)
    prob[nan_mask] = 0
    # d_theta = sum_q_[prob-correct]
    theta -= lr * np.sum(prob - zero_train_matrix, axis=1)
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))

    # update beta
    x = theta_mat - beta_mat
    prob = sigmoid(x) 
    prob[nan_mask] = 0
    # d_beta = sum_u_[prob-correct]
    beta -= lr * np.sum(zero_train_matrix - prob, axis=0)

    return theta, beta


def irt(data, train_matrix, zero_train_matrix, val_data, lr, iterations):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta)
    """
    theta = np.zeros(N)
    beta = np.zeros(D)

    # training and validation negtive log likelihoods
    train_neg_llds = []
    val_neg_llds = []
    # training and validation accuracies
    train_accs, val_accs = [], []

    for i in range(iterations):
        train_neg_llds.append(neg_log_likelihood(data, theta=theta, beta=beta))
        val_neg_llds.append(
            neg_log_likelihood(val_data, theta=theta, beta=beta))

        theta, beta = update_theta_beta(train_matrix, zero_train_matrix, lr, theta, beta)
        
        train_accs.append(evaluate(data=data, theta=theta, beta=beta))
        val_accs.append(evaluate(data=val_data, theta=theta, beta=beta))

    # plot training and validation negtive log likelihoods for each iteration
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_neg_llds, 'r')
    ax.plot(np.arange(iterations), val_neg_llds, 'g')
    ax.xaxis.set_label_text('iterations')
    ax.yaxis.set_label_text('negtive log likelihoods')
    ax.set_title('train and val negtive log likelihoods for each iteration')
    ax.legend(['train_neg_llds', 'val_neg_llds'])
    plt.savefig(
        './2b. train and val negtive log likelihoods for each iteration.png')

    # plot train and val accuracies for each iteration
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_accs, 'r')
    ax.plot(np.arange(iterations), val_accs, 'g')
    ax.xaxis.set_label_text('iteration')
    ax.yaxis.set_label_text('accuracies')
    ax.set_title('irt: train and val accuracies for each iteration')
    ax.legend(['train_accs', 'val_accs'])
    plt.savefig('./2b. train and val accuracies for each iteration.png')
    return theta, beta


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")

    train_matrix = load_train_sparse("../data").toarray()
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0

    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.003
    iterations = 150
    theta, beta = irt(train_data, train_matrix, zero_train_matrix, val_data, lr, iterations)
    
    # part (c) report the fianl validation and test accuracies
    final_val_acc = evaluate(data=val_data, theta=theta, beta=beta)
    final_test_acc = evaluate(data=test_data, theta=theta, beta=beta)
    print(
        f"The final validation accuracy is {final_val_acc}, the final test accuracy is {final_test_acc}")

    final_train_acc = evaluate(data=train_data, theta=theta, beta=beta)  
    print(f"The final training accuracy is {final_train_acc}")

    # part (d) select five questions and plot five curves showing p(c_uq)
    # as a function of theta given a question q
    q_list = random.choices(np.arange(D), k=5)
    plot_legends = []
    for i in range(5):
        plot_legends.append(f'question_id: {q_list[i]}')

    theta_range = np.linspace(-5, 5, num=101)

    curve_colors = ['r', 'g', 'b', 'c', 'y']
    fig, ax = plt.subplots()

    for i in range(5):
        q = q_list[i]
        # list of probabilities p(c_uq) for each theta given question q
        prob_list = sigmoid(theta_range-beta[q])
        # plot p(c_uq) as a function of theta given question q
        ax.plot(theta_range, prob_list, curve_colors[i])

    ax.xaxis.set_label_text('theta')
    ax.yaxis.set_label_text('p(c_ij)')
    ax.set_title(
        'p(c_ij) as a function of theta given five different questions')
    ax.legend(plot_legends)
    plt.savefig(
        './2d. p(c_ij) as a function of theta given given five different questions.png')


if __name__ == "__main__":
    main()
