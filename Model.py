import math
import numpy as np
import itertools
import pprint
import operator
from random import gauss
from collections import Counter


class Neuron:
    def __init__(self):
        self.value = 0


class Model:
    model_type = None
    model = None

    def __init__(self):
        pass

    def test(self):
        pass

    def train(self, data_point):
        pass


class Nearest(Model):
    vector_keys = []

    def __init__(self, indices_to_loop):
        self.model_type = "nearest"
        self.model = []
        self.vector_keys = indices_to_loop

    def train(self, train_row):
        self.model.append(train_row)

    def test(self, test_row):
        minimum_distance = float("Inf")
        current_classification = None
        for model_row in self.model:
            current_distance = self.distance(model_row, test_row)
            if current_distance < minimum_distance:
                minimum_distance = current_distance
                current_classification = model_row["orientation"]
        return test_row["id"], current_classification

    def distance(self, dp_1, dp_2):
        sum_distance = 0
        for index in self.vector_keys:
            sum_distance += math.pow(dp_2[index] - dp_1[index], 2)
        return sum_distance


class AdaBoost(Model):
    vector_keys = []
    stump_allocation = {"0": [], "90": [], "180": [], "270": []}

    def __init__(self, indices_to_loop, stump_count):
        self.model_type = "adaboost"
        self.model = []
        self.vector_keys = indices_to_loop
        self.stump_count = stump_count

    def train(self, train_rows):
        feature_combinations = [i for i in itertools.permutations(self.vector_keys, 2)]
        counter = 1
        for orientation in ["0", "90", "180", "270"]:
            print(orientation)
            stumps_left = self.stump_count
            weights = [1.0 / len(train_rows)] * len(train_rows)
            while stumps_left > 0:
                print("Next Stump")
                pprint.pprint(weights[1:30])
                performance_index = {}
                for current_combination in feature_combinations[0:100]:
                    if current_combination in self.stump_allocation[str(orientation)]:
                        continue
                    correct_counts = 0
                    incorrect_counts = 0
                    error_count = 0
                    all_count = 0
                    totals = 0
                    if counter % 1000 == 0:
                        print 1.0 * counter / 36672
                    train_index = 0
                    for weight_index in range(0, len(weights)):
                        totals += weights[weight_index]
                        if str(train_rows[weight_index]["orientation"]) == orientation:
                            all_count += 1
                            if train_rows[weight_index][current_combination[0]] > train_rows[weight_index][
                                current_combination[1]]:
                                correct_counts += weights[weight_index]
                            else:
                                incorrect_counts += weights[weight_index]
                                error_count += 1
                        train_index += 1
                    current_performance = (1.0 * correct_counts) / totals
                    performance_index[(current_combination[0], current_combination[1])] = current_performance
                current_stump_features = max(performance_index.iteritems(), key=operator.itemgetter(1))[0]
                error = error_count / all_count
                print("Selected features")
                print(current_stump_features)
                print(error)
                self.stump_allocation[str(orientation)].append(current_stump_features)
                for weight_index in range(0, len(weights)):
                    if str(train_rows[weight_index]["orientation"]) == orientation:
                        if train_rows[weight_index][current_stump_features[0]] < train_rows[weight_index][
                            current_stump_features[1]]:
                            weights[weight_index] *= (error / (1.0 - error))
                base = sum(weights)
                for weight_index in range(0, len(weights)):
                    weights[weight_index] = weights[weight_index] / base
                stumps_left -= 1

    def __str__(self):
        return pprint.pformat(self.stump_allocation)


class NNet(Model):
    def __init__(self, hidden_nodes, length):
        self.model_type = "nnet"
        self.length = length
        self.h_weights = {}
        self.o_weights = {}
        self.hidden_nodes = hidden_nodes
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        for i in range(0, length):
            temp_neuron = Neuron()
            self.input_neurons.append(temp_neuron)

        for i in range(0, hidden_nodes):
            temp_neuron = Neuron()
            self.hidden_neurons.append(temp_neuron)

        for i in range(0, 4):
            temp_neuron = Neuron()
            self.output_neurons.append(temp_neuron)

    def get_rnd(self):
        return 0.1

    def step_function(self, x):
        if x > 0:
            return x
        return 0

    def result_map(self, x):
        value = {0: [1, 0, 0, 0], 90: [0, 1, 0, 0], 180: [0, 0, 1, 0], 270: [0, 0, 0, 1]}
        return value[x]


    def soft_max(self, x):
        max_element = max(x)
        x = [(inp - max_element) for inp in x]
        sum_element = float(sum([math.e ** inp for inp in x]))
        ret_res = [math.e ** inp / sum_element for inp in x]
        return ret_res

    def generate_gaussian(self):
        return int(gauss(float(255 / 2), float(255 / 4)))

    def train(self, train_row):
        self.model = train_row

        # create a list for weights for the hidden layer
        for i, input_item in enumerate(self.input_neurons):
            for j, hidden_item in enumerate(self.hidden_neurons):
                if i not in self.h_weights:
                    self.h_weights[i] = {}
                self.h_weights[i][j] = self.generate_gaussian()

        # create a list of weights for the output layer
        for i, hidden_item in enumerate(self.input_neurons):
            for j, output_item in enumerate(self.output_neurons):
                if i not in self.o_weights:
                    self.o_weights[i] = {}
                self.o_weights[i][j] = self.generate_gaussian()

        # exp_output = [0, 90, 180, 270]

        # Feed Forward Network
        # --------------------

        ##### Input Layer #########
        # assign the neurons in the input layer with a value
        for train_item in train_row:
            for index, value in enumerate(train_item[2:]):
                self.input_neurons[index].value = value

            ##### Hidden Layer ########
            # Apply the step function on the Sum of ( output of input layer * hidden weight list )
            for j, hidden_item in enumerate(self.hidden_neurons):
                total = 0
                for i, input_item in enumerate(self.input_neurons):
                    total += (input_item.value * self.h_weights[i][j])

                # apply step function on the sum
                total = self.step_function(total)
                hidden_item.value = total

            ##### Output Layer ########
            # Apply the step function on the Sum of ( output of hidden layer * output weight list )
            for j, output_item in enumerate(self.output_neurons):
                total = 0
                for i, hidden_item in enumerate(self.input_neurons):
                    total += (hidden_item.value * self.o_weights[i][j])
                total = self.step_function(total)
                output_item.value = total

            # output prediction for the orientation in the form [0, 90, 180, 270]
            maximum = self.soft_max([x.value for x in self.output_neurons])

            # print maximum
            # raw_input()
            # for x in self.output_neurons:
            #     if maximum != x.value:
            #         x.value = 0
            #     else:
            #         x.value = 1

            # for x in self.output_neurons:
            # print x.value

            # Backpropogation Starts here
            # ---------------------------

            # calculate output Delta
            output_delta = {index: x - self.output_neurons[index].value for index, x in
                            enumerate(self.result_map(train_item[1]))}
            # print self.result_map(train_item[1])
            # print output_delta


            # calculate hidden Delta
            hidden_delta = {}
            for i, hidden_item in enumerate(self.hidden_neurons):
                total = 0
                for j, output_item in enumerate(self.output_neurons):
                    total += (output_delta[j] * self.o_weights[i][j])
                    # print total, output_delta[j], self.o_weights[i][j]
                    # if math.isnan(total):
                    #     raw_input()
                hidden_delta[i] = total

            # Applying Weights
            alpha = 0.0000000000001
            for j, hidden_item in enumerate(self.hidden_neurons):
                for i, input_item in enumerate(self.input_neurons):
                    self.h_weights[i][j] += (alpha * input_item.value * hidden_delta[j])

            for j, output_item in enumerate(self.output_neurons):
                for i, hidden_item in enumerate(self.hidden_neurons):
                    self.o_weights[i][j] += (alpha * hidden_item.value * output_delta[j])
        #print self.h_weights
        print self.o_weights
        print "Training Complete"

    def get_orientation(self, x):
        values = {0: 0, 1: 90, 2: 180, 3: 270}
        return values[x.index(max(x))]

    def test(self, train_row):
        values = Counter()
        correct = 0
        incorrect = 0
        ##### Input Layer #########
        # assign the neurons in the input layer with a value
        for train_item in train_row:
            for index, value in enumerate(train_item[2:]):
                self.input_neurons[index].value = value

            ##### Hidden Layer ########
            # Apply the step function on the Sum of ( output of input layer * hidden weight list )
            for j, hidden_item in enumerate(self.hidden_neurons):
                total = 0
                for i, input_item in enumerate(self.input_neurons):
                    total += (input_item.value * self.h_weights[i][j])

                # apply step function on the sum
                total = self.step_function(total)
                hidden_item.value = total

            ##### Output Layer ########
            # Apply the step function on the Sum of ( output of hidden layer * output weight list )
            for j, output_item in enumerate(self.output_neurons):
                total = 0
                for i, hidden_item in enumerate(self.input_neurons):
                    total += (hidden_item.value * self.o_weights[i][j])
                total = self.step_function(total)
                output_item.value = total

            # output prediction for the orientation in the form [0, 90, 180, 270]
            maximum = self.soft_max([x.value for x in self.output_neurons])
            if int(train_item[1]) == self.get_orientation(maximum):
                 values[self.get_orientation(maximum)] += 1


        print "correct : ",values

