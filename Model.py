import math
import itertools
import pprint
import operator


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
    def __init__(self):
        self.model_type = "nnet"
