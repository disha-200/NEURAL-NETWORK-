import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self):
        self.layers = [9, 2, 4, 2]
        self.weight_list = []
        self.activation_list = [None] * len(self.layers)
        self.delta_list = [None] * len(self.layers)
        self.cost_list_train = []
        self.cost_list_test = []


    def generate_weight(self, no_attribute):
        for i in range(len(self.layers) - 1):
            rows = self.layers[i + 1]
            if i == 0:
                columns = no_attribute + 1  # Increase the number of columns by 1
            else:
                columns = self.layers[i] + 1
            weight_matrix = np.random.uniform(low=-1, high=1, size=(rows, columns))
            weight_matrix[weight_matrix == 0] = np.random.uniform(low=0, high=1,
                                                                  size=(np.count_nonzero(weight_matrix == 0),))
            self.weight_list.append(weight_matrix)

    def sumsquare_weight(self):
        sum_of_squares = 0
        for weight_matrix in self.weight_list:
            sum_of_squares += np.sum(np.square(weight_matrix[:, 1:]))
        return sum_of_squares


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propogation(self, x):
        a = x
        for k in range(1, len(self.layers) - 1):  #range(1, len(self.layers)-1)
            if k == 1:
                a_prev = np.insert(a, 0, 1, axis=0)
                self.activation_list[0] = a_prev # Include 1 as the first element of x
            weight_matrix = self.weight_list[k - 1]
            z = np.dot(weight_matrix, self.activation_list[k-1])
            a = self.sigmoid(z)
            a = np.insert(a, 0, 1, axis=0)
            self.activation_list[k] = a
        penultimate_a = self.activation_list[len(self.layers) - 2]
        last_wts = self.weight_list[-1]
        z_final = np.dot(last_wts, penultimate_a)
        a_final = self.sigmoid(z_final)
        self.activation_list[-1] = a_final
        return a_final

    def cost_function(self, X, Y, regularizer_lambda):
        total_cost = 0
        for i in range(len(X)):
            x = np.array(X[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape
            # as a column vector
            y = np.array(Y[i]).reshape(-1, 1)

            if y == 0:
                y = np.array([[1], [0]])
            else:
                y = np.array([[0], [1]])

            f_x = self.forward_propogation(x)
            j = (-y * np.log(f_x)) - ((1-y) * np.log(1-f_x))
            total_cost = total_cost + np.sum(j)
        total_cost = total_cost / len(X)
        s = self.sumsquare_weight()
        s = (regularizer_lambda/(2 * len(X))) * s

        return total_cost+s



    def back_propogation(self, X_train, y_train, X_test, y_test, iterations, regularizer_lambda, alpha):
        gradient_list = []
        for j in range(len(self.layers) - 1):
            rows = self.layers[j + 1]
            columns = self.layers[j] + 1
            gradient_list.append(np.zeros((rows, columns)))
        for n in range(iterations):
            for i in range(len(X_train)):
                x = np.array(X_train[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape as a column vector
                y = np.array(y_train[i]).reshape(-1, 1)
                # print(y.shape,"shape before")
                # print(y, "y after reshape")

                if y == [[0]]:
                    # print("0 hi")
                    y = np.array([[1], [0]])
                else:
                    # print("hi")
                    y = np.array([[0], [1]])
                # print(y.shape, "shape after")
                f_x = self.forward_propogation(x)

                delta_initial = f_x - y
                self.delta_list[len(self.layers)-1] = delta_initial

                for k in range(len(self.layers) - 2, 0, -1):  # range(len(self.layers) - 1, 0, -1), only for hidden layers
                    theta_transpose = np.transpose(self.weight_list[k])
                    # if activation is stored
                    if k == len(self.layers) - 2:
                        delta = np.dot(theta_transpose, delta_initial) * self.activation_list[k] * (1 - self.activation_list[k])
                        self.delta_list[k] = delta[1:]
                    else:
                        delta = np.dot(theta_transpose, self.delta_list[k+1]) * self.activation_list[k] * (1 - self.activation_list[k])
                        self.delta_list[k] = delta[1:]

                for k in range(len(self.layers) - 2, -1, -1):
                    activation_transpose = np.transpose(self.activation_list[k])
                    # gradient[k] = gradient[k] + self.delta_list[k+1] * activation_transpose
                    gradient_list[k] += self.delta_list[k + 1] * activation_transpose
                    # self.gradient_list[k] = gradient

            for k in range(len(self.layers) - 2, -1, -1):
                p = regularizer_lambda * self.weight_list[k]
                p[:, 0] = 0
                gradient_list[k] = (1/len(X_train))*(gradient_list[k] + p)

            for k in range(len(self.layers) - 2, -1, -1):
                self.weight_list[k] = self.weight_list[k] - alpha*gradient_list[k]

            cost_output_train = network.cost_function(X_train, y_train, regularizer_lambda=0.05)
            cost_output_test = network.cost_function(X_test, y_test, regularizer_lambda=0.05)
            self.cost_list_train.append(cost_output_train)
            self.cost_list_test.append(cost_output_test)

    def predict(self, X, y):
        pred_list = []
        actual_list = []

        for i in range(len(X)):
            x = np.array(X[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape
            # as a column vector
            actual = y[i]

            f_x = self.forward_propogation(x)

            if f_x[0] > f_x[1]:
                pred = 0
            else:
                pred = 1

            pred_list.append(pred)
            actual_list.append(actual)

        accuracy, f1 = self.evaluate(actual_list, pred_list)
        return accuracy, f1

    def evaluate(self, y_true, y_pred):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if y_true[i] == 1:
                    false_negatives += 1
                else:
                    false_positives += 1
        accuracy = (true_positives + true_negatives) / len(y_true)
        if true_positives + false_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, f1


def n(d):
    for col in d.columns:
        d[col] = (d[col]-d[col].min()) / (d[col].max() - d[col].min())
    return d


data = pd.read_csv('hw3_cancer.csv', delimiter='\t')
data = data.rename(columns={'Class': 'target'})
target_column = data.iloc[:, -1]
data_wo_target = data.drop(labels='target',axis=1,inplace=False)
ndata = n(data_wo_target)
data = pd.concat([ndata, target_column], axis=1)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36, shuffle=True)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

network = NeuralNetwork()
no_attribute = X_train.shape[1]
network.generate_weight(no_attribute)

network.back_propogation(X_train, y_train, X_test, y_test, iterations=500, regularizer_lambda=0.05, alpha=0.5)
cost_list_train = network.cost_list_train
cost_list_test = network.cost_list_test
# cost_output = network.cost_function(X_train, y_train, regularizer_lambda=0.00)

plt.plot(range(1, 501), cost_list_train)
# plt.xticks(range(1, 501, 50))
plt.title('Cost vs. Iterations')
plt.xlabel('Number of Iteration * 559 instances')
plt.ylabel('Cost over training set')
plt.show()

plt.plot(range(1, 501), cost_list_test)
# plt.xticks(range(1, 501, 50))
plt.title('Cost vs. Iterations')
plt.xlabel('Number of Iteration * 559 instances')
plt.ylabel('Cost over testing set')
plt.show()





