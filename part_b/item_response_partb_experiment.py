from utils import *

import numpy as np
import matplotlib.pyplot as plt

# size of students, questions and subjects
N, D, S = 542, 1774, 388


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, c, k):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: N x S matrix
    :param c: float
    :param k: Vector
    :param q_meta: D x S matrix
    :return: float
    """
    log_lklihood = 0

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]

        x = k[q] * (theta[u] - beta[q])
        prob = sigmoid(x) * (1 - c) + c

        correct = data["is_correct"][i]
        log_lklihood += correct * np.log(prob) + (1 - correct) * np.log(
            (1 - prob))
    return -log_lklihood


def update_parameters(train_matrix, zero_train_matrix, lr, theta, beta,
                      c, k, lambd):
    """ Update theta, beta alpha, c and k using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: N x S matrix
    :param c: float
    :param k: Vector
    :param lambd: float
    :param q_meta: D x S matrix
    :return: tuple of vectors
    """
    # create matrix representation for calculation convenience
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)
    k_mat = np.ones((N, 1)) @ np.expand_dims(k, axis=0)

    nan_mask = np.isnan(train_matrix)

    # update theta
    x = (theta_mat - beta_mat) * k_mat
    prob = sigmoid(x) * (1 - c) + c
    prob[nan_mask] = 0
    theta -= lr * (np.sum((prob - zero_train_matrix) * k_mat, axis=1) * (
                1 - c) + lambd * theta)
    theta_mat = np.expand_dims(theta, axis=1) @ np.ones((1, D))

    # update beta
    x = (theta_mat - beta_mat) * k_mat
    prob = sigmoid(x) * (1 - c) + c
    prob[nan_mask] = 0
    beta -= lr * (np.sum((zero_train_matrix - prob) * k_mat, axis=0) * (
                1 - c) + lambd * beta)
    beta_mat = np.ones((N, 1)) @ np.expand_dims(beta, axis=0)

    # update alpha
    x = (theta_mat - beta_mat) * k_mat
    prob = sigmoid(x) * (1 - c) + c
    prob[nan_mask] = 0

    # update k
    x = (theta_mat - beta_mat) * k_mat
    prob = sigmoid(x) * (1 - c) + c
    prob[nan_mask] = 0
    k -= lr * (np.sum(
        (prob - zero_train_matrix) * (theta_mat - beta_mat),
        axis=0) * (1 - c) + lambd * k)
    k_mat = np.ones((N, 1)) @ np.expand_dims(k, axis=0)

    # update c
    x = (theta_mat - beta_mat) * k_mat
    prob = sigmoid(x) * (1 - c) + c
    prob[nan_mask] = 0
    c -= lr * (1 - sigmoid(x).mean())
    # constrain c in range [0, 1]
    c = max(0, c)
    c = min(1, c)

    return theta, beta, c, k


def irt(data, train_matrix, zero_train_matrix, val_data, lr, iterations,
        lambd):
    """ Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param train_matrix: N x D matrix
    :param zero_train_matrix: N x D matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param alpha: N x S matrix
    :param lambd: float
    :param q_meta: D x S matrix
    :return: (theta, beta, alpha, c, k)
    """
    theta = np.ones(N) * 0.5
    beta = np.ones(D) * 0.5
    c = 0
    k = np.ones((D,)) * 0.5

    # training and validation negtive log likelihoods
    train_neg_llds, val_neg_llds = [], []
    # training and validation accuracies
    train_accs, val_accs = [], []

    for i in range(iterations):

        train_neg_llds.append(
            neg_log_likelihood(data, theta, beta, c, k))
        val_neg_llds.append(
            neg_log_likelihood(val_data, theta, beta, c, k))

        theta, beta, c, k = update_parameters(train_matrix,
                                            zero_train_matrix, lr,
                                            theta, beta, c, k,
                                            lambd)

        val_acc = evaluate(val_data, theta, beta, c, k)
        train_acc = evaluate(data, theta, beta, c, k)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print("Epoch: {}/{} \t "
              "Train Acc: {}\t"
              "Valid Acc: {}".format(i, iterations, train_acc, val_acc))

    # plot training and validation negtive log likelihoods for each iteration
    plt.figure(1)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_neg_llds, 'r')
    ax.plot(np.arange(iterations), val_neg_llds, 'g')
    ax.xaxis.set_label_text('iterations')
    ax.yaxis.set_label_text('negtive log likelihoods')
    ax.set_title(
        'irt_experiment: train and val negtive log likelihoods '
        'for each iteration')
    ax.legend(['train_neg_llds', 'val_neg_llds'])
    plt.savefig(
        'partb_experiment.png')

    # plot train and val accuracies for each iteration
    plt.figure(2)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), train_accs, 'r')
    ax.plot(np.arange(iterations), val_accs, 'g')
    ax.xaxis.set_label_text('iteration')
    ax.yaxis.set_label_text('accuracies')
    ax.set_title('irt_experiment: train and val accuracies for each iteration')
    ax.legend(['train_accs', 'val_accs'])
    plt.savefig(
        'parb_experiment2.png')

    return theta, beta, c, k


def evaluate(data, theta, beta, c, k):
    """ Evaluate the model given data and return the accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: N x S matrix
    :param c: float
    :param k: Vector
    :param q_meta: D x S matrix
    :return: float
    """
    pred = []

    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] - beta[q]) * k[q]).sum()
        p_a = sigmoid(x) * (1 - c) + c
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

    lr = 0.02
    lambd = 0.1
    iterations = 60
    theta, beta, c, k = irt(train_data, train_matrix, zero_train_matrix,
                                   val_data, lr, iterations, lambd)

    # part (c) report the fianl validation and test accuracies
    final_val_acc = evaluate(val_data, theta, beta, c, k)
    final_test_acc = evaluate(test_data, theta, beta, c, k)
    print(
        f"The final validation accuracy is {final_val_acc}, "
        f"the final test accuracy is {final_test_acc}")

    final_train_acc = evaluate(train_data, theta, beta, c, k)
    print(f"The final training accuracy is {final_train_acc}")


if __name__ == "__main__":
    main()
