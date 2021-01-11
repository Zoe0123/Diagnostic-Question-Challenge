from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch
import ast

import matplotlib.pyplot as plt


N, D, S = 542, 1774, 388

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class Encoder(nn.Module):
    def __init__(self, num_question, k_1, k_2, dropout_rate):
        super(Encoder, self).__init__()
        self.layer_1 = nn.Linear(num_question, k_1)
        # self.layer_2 = nn.Linear(k_1, k_2)
        self.dropout = nn.Dropout(dropout_rate)

    def get_weight_norm(self):
        """ Return ||W||

        :return: float
        """
        layer_1_w_norm = torch.norm(self.layer_1.weight, 2)
        # layer_2_w_norm = torch.norm(self.layer_2.weight, 2)
        # return layer_1_w_norm + layer_2_w_norm
        return layer_1_w_norm

    def forward(self, inputs):
        out = inputs
        out = self.dropout(out)
        out = self.layer_1(out)
        # out = F.relu(out)
        # out = self.layer_2(out)
        out = F.sigmoid(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_question, k_3, k_4, dropout_rate):
        super(Decoder, self).__init__()
        # self.layer_1 = nn.Linear(k_3, k_4)
        self.layer_2 = nn.Linear(k_4, num_question)
        self.dropout = nn.Dropout(dropout_rate)

    def get_weight_norm(self):
        """ Return ||W||

        :return: float
        """
        # layer_1_w_norm = torch.norm(self.layer_1.weight, 2)
        layer_2_w_norm = torch.norm(self.layer_2.weight, 2)
        # return layer_1_w_norm + layer_2_w_norm
        return layer_2_w_norm

    def forward(self, inputs):
        out = inputs
        out = self.dropout(out)
        # out = self.layer_1(out)
        # out = F.tanh(out)
        out = self.layer_2(out)
        out = F.sigmoid(out)
        return out


class VAE(nn.Module):

    def __init__(self, encoder, decoder, k_2, k_3):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(k_2, k_3)
        self._enc_log_sigma = torch.nn.Linear(k_2, k_3)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        return self.encoder.get_weight_norm() + self.decoder.get_weight_norm()

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False) 

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self.sample_latent(h_enc)
        return self.decoder(z)


def kl_divergence(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, beta, alpha):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_student, num_question = train_data.shape

    # list to record change in learning objective
    train_accs = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        incorrect_guess = 0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs) + alpha[user_id]

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = (
                torch.sum((output - target) ** 2.) + 
                lamb * model.get_weight_norm() / 2 + 
                beta * kl_divergence(model.z_mean, model.z_sigma)
            )
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            guess = (output >= 0.5).long()
            target_label = (target >= 0.5).long()
            incorrect_guess += torch.sum((guess - target_label)**2)

        total_train_example = num_student * num_question - np.isnan(train_data.unsqueeze(0).numpy()).sum()
        train_acc = 1 - int(incorrect_guess) / total_train_example
        train_accs.append(train_acc)
        valid_acc = evaluate(model, zero_train_data, valid_data, alpha)
        valid_accs.append(valid_acc)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Train Acc: {}\t"
              "Valid Acc: {}".format(epoch, train_loss, train_acc, valid_acc))

    epochs = list(range(num_epoch))
    plt.plot(epochs, train_accs, label = "training accuracy")
    plt.plot(epochs, valid_accs, label = "validation accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training & Validation Accuracy vs. Epoch (k = 50)')
    plt.legend()
    plt.savefig("vae_accuracy.png")


def evaluate(model, train_data, valid_data, alpha):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs) + alpha[u]

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def load_qmeta(base_path="../data"):
    """ Load the question meta data.
    :return: q_meta: D X S matrix, each entry 0/1 represents whether the 
    subject is included in this question.
    """
    path = os.path.join(base_path, "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))

    # Initialize the data.
    q_meta = np.zeros((D, S))
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                question_id = int(row[0])
                subject = ast.literal_eval(row[1])
                q_meta[question_id][subject] = 1
            except ValueError:
                # Pass first row.
                pass

    return q_meta

def get_alpha(q_meta, train_matrix, zero_train_matrix):
    """ Get the alpha matrix, with entries represents for each student, the 
    average familiarity on the subjects included by each question.
    :return: alpha: N x D matrix
    """
    # for each student, count the number of correctness on each subject
    sub_correct = zero_train_matrix @ q_meta
    
    train_matrix_copy = train_matrix.clone()
    # num_subject: NxS matrix: number of questions including each subject, answered by each student
    train_matrix_copy[train_matrix_copy == 0] = 1
    train_matrix_copy[np.isnan(train_matrix_copy)] = 0
    num_subject = train_matrix_copy @ q_meta
    # to prevent devide by 0
    num_subject[num_subject==0] = 1

    # familarity of each student on each subject (fraction of questions 
    # contained this subject answered correctly by this student, based on data already known)
    sub_familiar = sub_correct / num_subject
    
    # for each student, the avg familiarity on the subjects included by each question.
    alpha = (sub_familiar @ q_meta.T) / np.count_nonzero(q_meta, axis=1)
    #alpha = (alpha.T / np.amin(alpha, axis=1)).T
    return alpha


if __name__ == '__main__':

    zero_train_matrix, train_matrix, valid_data, test_data = load_data() 
    
    k_1, k_2, k_3, k_4 = 20, 20, 15, 15
    dropout_rate_1, dropout_rate_2 = 0.1, 0.1
    
    encoder = Encoder(num_question=train_matrix.shape[1], k_1=k_1, k_2=k_2, dropout_rate=dropout_rate_1)
    decoder = Decoder(num_question=train_matrix.shape[1], k_3=k_3, k_4=k_4, dropout_rate=dropout_rate_2)
    model = VAE(encoder, decoder, k_2=k_2, k_3=k_3)

    # Set optimization hyperparameters.
    lr = 0.01
    lamb = 0.15
    num_epoch = 10
    beta = 0.3

    # load question_mata.csv 
    q_meta = load_qmeta()
    # alpha: for each student, the avrage familiarity on the subjects included by each question.
    alpha = get_alpha(q_meta, train_matrix, zero_train_matrix).float() * 0.2

    # train the model
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch, beta, alpha)
    
    # final test accuracy
    final_test_acc = evaluate(model, zero_train_matrix, test_data, alpha)
    print(f"final test accracy is {final_test_acc}")
