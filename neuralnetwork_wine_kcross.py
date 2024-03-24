import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self):
        self.layers = [13, 16, 3]
        self.weight_list = []
        self.activation_list = [None] * len(self.layers)
        self.delta_list = [None] * len(self.layers)


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
        for k in range(1, len(self.layers) - 1):# range(1, len(self.layers)-1)
            # print("loop val", k)
            # print("loop ", k)
            if k == 1:
                a_prev = np.insert(a, 0, 1, axis=0)
                self.activation_list[0] = a_prev # Include 1 as the first element of x
                # print("input activation", self.activation_list[0])
            # print("wt and acti layer index of previous", k-1)
            weight_matrix = self.weight_list[k - 1]
            z = np.dot(weight_matrix, self.activation_list[k-1])
            # print("z ", z)
            a = self.sigmoid(z)
            a = np.insert(a, 0, 1, axis=0)
            self.activation_list[k] = a
            # print("a calculated for loop", self.activation_list[k])
        # print("penultimate index", len(self.layers) - 2)
        penultimate_a = self.activation_list[len(self.layers) - 2]
        last_wts = self.weight_list[len(self.layers) - 2]
        z_final = np.dot(last_wts, penultimate_a)
        a_final = self.sigmoid(z_final)
        # print("final z", z_final)
        # print("final a", a_final)
        self.activation_list[-1] = a_final

        # print("wt list", self.weight_list)
        # print("acti list", self.activation_list)
        return a_final

    def cost_function(self, X, Y, regularizer_lambda):
        total_cost = 0
        for i in range(len(X)):
            x = np.array(X[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape
            # as a column vector
            y = np.array(Y[i]).reshape(-1, 1)

            if y == [[1]]:
                # print("0 hi")
                y = np.array([[1], [0], [0]])
            else:
                if y == [[2]]:
                    # print("hi")
                    y = np.array([[0], [1], [0]])
                else:
                    y = np.array([[0], [0], [1]])

            f_x = self.forward_propogation(x)
            j = (-y * np.log(f_x)) - ((1-y) * np.log(1-f_x))
            total_cost = total_cost + np.sum(j)
        total_cost = total_cost / len(X)
        s = self.sumsquare_weight()
        s = (regularizer_lambda / (2 * len(X))) * s

        return total_cost+s


    def back_propogation(self, X_train, y_train, iterations, regularizer_lambda, alpha):
        gradient_list = []
        for j in range(len(self.layers) - 1):
            rows = self.layers[j + 1]
            columns = self.layers[j] + 1
            gradient_list.append(np.zeros((rows, columns)))
        for n in range(iterations):
            for i in range(len(X_train)):
                x = np.array(X_train[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape as a column vector
                y = np.array(y_train[i]).reshape(-1, 1)

                # print("---------------------------------------------------")
                # print("instance ", i)
                # print("x instanxe", x)
                # # print(y.shape,"shape before")
                # # print(y, "y after reshape")
                # print(y, "y value")
                if y == [[1]]:
                    # print("0 hi")
                    y = np.array([[1], [0], [0]])
                else:
                    if y == [[2]]:
                        # print("hi")
                        y = np.array([[0], [1], [0]])
                    else:
                        y = np.array([[0], [0], [1]])

                # print("y instance", y)
                # print(y, "after explicit assignment")
                # print(y.shape, "shape after")
                f_x = self.forward_propogation(x)
                # print("f_x", f_x)
                delta_initial = f_x - y
                self.delta_list[len(self.layers)-1] = delta_initial
                # print("delta inital ", len(self.layers)-1, " ", delta_initial)

                # print("delta loop values")
                for k in range(len(self.layers) - 2, 0, -1): # range(len(self.layers) - 1, 0, -1), only for
                    # hidden layers
                    # print(k)
                    theta_transpose = np.transpose(self.weight_list[k])
                    # if activation is stored
                    if k == len(self.layers) - 2:
                        delta = np.dot(theta_transpose, delta_initial) * self.activation_list[k] * (1 - self.activation_list[k])
                        self.delta_list[k] = delta[1:]
                        # print("delta ", k, " ", self.delta_list[k])
                    else:
                        delta = np.dot(theta_transpose, self.delta_list[k+1]) * self.activation_list[k] * (1 - self.activation_list[k])
                        self.delta_list[k] = delta[1:]
                        # print("delta ", k, " ", self.delta_list[k])

                # print("gradient loop values")
                for k in range(len(self.layers) - 2, -1, -1):
                    # print(k)
                    activation_transpose = np.transpose(self.activation_list[k])
                    # print("activation trans ", activation_transpose)
                    # print("delta of next layer ", self.delta_list[k + 1])
                    # gradient[k] = gradient[k] + self.delta_list[k+1] * activation_transpose
                    gradient_list[k] += self.delta_list[k + 1] * activation_transpose
                    # print("gradient of", k, " ", self.delta_list[k + 1] * activation_transpose)
                    # self.gradient_list[k] = gradient
            # print("--------------------------------------------------------")
            # print("reg update loop values")
            for k in range(len(self.layers) - 2, -1, -1):
                # print(k)
                p = regularizer_lambda * self.weight_list[k]
                p[:, 0] = 0
                gradient_list[k] = (1/len(X_train))*(gradient_list[k] + p)
                # print("reg gradient of", k, " ", gradient_list[k])

            # print("wt update loop values")
            for k in range(len(self.layers) - 2, -1, -1):
                # print(k)
                self.weight_list[k] = self.weight_list[k] - alpha*gradient_list[k]

            # cost_output = network.cost_function(X_train, y_train, regularizer_lambda=0.00)
            # print(cost_output, " cost func output for each ", n)

    def predict(self, X, y):
        pred_list = []
        actual_list = []

        for i in range(len(X)):
            x = np.array(X[i]).reshape(-1, 1)  # Get the first training instance, convert to numpy array and reshape
            # as a column vector
            actual = y[i]

            f_x = self.forward_propogation(x)
            # print(f_x,"fx")

            max_index = np.argmax(f_x)
            # print("max index", max_index)

            pred = max_index+1  # since classes are 1, 2 and 3

            pred_list.append(pred)
            actual_list.append(actual)

        # print(pred_list, "preds")
        # print(actual_list, "actuals")

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
        if true_positives + false_positives == 0 or true_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives == 0 or true_positives == 0:
            recall = 1
        else:
            recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * (precision * recall) / (precision + recall)
        return accuracy, f1


def n(d):
    for col in d.columns:
        d[col] = (d[col]-d[col].min()) / (d[col].max() - d[col].min())
    return d


def K_fold_datasets(dataset, k):
    class_frame = dataset.groupby('target')
    label_folds = {}

    for label, rows in class_frame:
        # Divide rows into k folds
        folds = np.array_split(rows, k)

        # Add folds to dictionary
        label_folds[label] = folds

    # Combine folds across labels
    all_folds = []
    for i in range(k):
        fold = []
        for label in label_folds:
            fold_data = label_folds[label][i]
            if isinstance(fold_data, str):
                fold_data = pd.DataFrame([fold_data])
            fold.append(fold_data)
        all_folds.append(pd.concat(fold))

    return all_folds


data = pd.read_csv('hw3_wine.csv', delimiter='\t')
data = data.rename(columns={'# class': 'target'})
target_column = data.iloc[:, 0]
data_wo_target = data.drop(labels='target', axis=1, inplace=False)
ndata = n(data_wo_target)
data = pd.concat([ndata, target_column], axis=1)
k = 10
full_folds = K_fold_datasets(data, k)

train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1 = [], [], \
    [], [], [], [], [], []
for i in range(k):
    # print(i+1, " fold")
    current_test_fold = full_folds[i]
    current_train_folds = pd.DataFrame()
    for j in range(k):
        if j != i:
            current_train_folds = pd.concat([current_train_folds, full_folds[j]])

    X_train = current_train_folds.iloc[:, :-1].values
    y_train = current_train_folds.iloc[:, -1].values

    X_test = current_test_fold.iloc[:, :-1].values
    y_test = current_test_fold.iloc[:, -1].values

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    network = NeuralNetwork()
    no_attribute = X_train.shape[1]
    network.generate_weight(no_attribute)

    network.back_propogation(X_train, y_train, iterations=500, regularizer_lambda=0.05, alpha=0.25)
    # cost_output = network.cost_function(X_train, y_train, regularizer_lambda=0.00)

    accuracy_train, f1_train = network.predict(X_train, y_train)
    train_accuracy.append(accuracy_train)
    # train_precision.append(precision_train)
    # train_recall.append(recall_train)
    train_f1.append(f1_train)
    # print(accuracy_train, " train accuracy")

    accuracy_test, f1_test = network.predict(X_test, y_test)
    test_accuracy.append(accuracy_test)
    # test_precision.append(precision_test)
    # test_recall.append(recall_test)
    test_f1.append(f1_test)
    # print(accuracy_test, " test accuracy")

mean_train_accuracy = np.mean(train_accuracy)
# mean_train_precision = np.mean(train_precision)
# mean_train_recall = np.mean(train_recall)
mean_train_f1 = np.mean(train_f1)

mean_test_accuracy = np.mean(test_accuracy)
# mean_test_precision = np.mean(test_precision)
# mean_test_recall = np.mean(test_recall)
mean_test_f1 = np.mean(test_f1)

print("train accuracy", mean_train_accuracy)
# print("train precision", mean_train_precision)
# print("train recall", mean_train_recall)
print("train f1", mean_train_f1)

print("test accuracy", mean_test_accuracy)
# print("test precision", mean_test_precision)
# print("test recall", mean_test_recall)
print("test f1", mean_test_f1)


# X_train = data.iloc[:, :-1].values
# y_train = data.iloc[:, -1].values
# network = NeuralNetwork()
# no_attribute = X_train.shape[1]
# network.generate_weight(no_attribute)
#
# network.back_propogation(X_train, y_train, iterations=500, regularizer_lambda=1, alpha=1)
# # cost_output = network.cost_function(X_train, y_train, regularizer_lambda=0.00)
#
# accuracy = network.predict(X_train, y_train)
# print(accuracy, " accuracy")
