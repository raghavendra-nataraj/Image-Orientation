import math
import itertools
import pprint
import operator
import csv
import time


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
        rows_considered = train_rows
        feature_combinations = [i for i in itertools.permutations(self.vector_keys, 2)]
        counter = 1
        incorrect_ids = []
        past_incorrect_ids = []
        start = time.time()
        for orientation in ["0", "90", "180", "270"]:
            print("New orientation:" + orientation)
            stumps_left = self.stump_count
            weights = [1.0 / len(rows_considered)] * len(rows_considered)
            while stumps_left > 0:
                incorrect_ids = []
                # pprint.pprint(weights[1:30])
                performance_index = {}
                counter = 0
                for current_combination in feature_combinations:
                    if current_combination in self.stump_allocation[str(orientation)]:
                        print("Skipping existing combination" + str(current_combination))
                        continue
                    correct_counts = 0
                    incorrect_counts = 0
                    error_count = 0
                    all_count = 0
                    totals = 0
                    if counter % 1000 == 0:
                        end = time.time()
                        print str((1.0 * counter) / 36672) + " at " + str(end - start)
                    train_index = 0
                    for weight_index in range(0, len(weights)):
                        totals += weights[weight_index]
                        if str(rows_considered[weight_index]["orientation"]) == orientation:
                            all_count += 1
                            if rows_considered[weight_index][current_combination[0]] > rows_considered[weight_index][
                                current_combination[1]]:
                                correct_counts += weights[weight_index]
                            else:
                                incorrect_counts += weights[weight_index]
                                error_count += 1
                        train_index += 1
                    current_performance = (1.0 * correct_counts) / totals
                    performance_index[(current_combination[0], current_combination[1])] = current_performance
                    counter += 1
                current_stump_features = max(performance_index.iteritems(), key=operator.itemgetter(1))[0]
                error = (1.0 * error_count) / all_count
                error_count = 0
                self.stump_allocation[str(orientation)].append(current_stump_features)
                self.save("in_progress_model.model")
                for weight_index in range(0, len(weights)):
                    if str(rows_considered[weight_index]["orientation"]) == orientation:
                        if rows_considered[weight_index][current_stump_features[0]] < rows_considered[weight_index][
                            current_stump_features[1]]:
                            if len(past_incorrect_ids) < 1 or weight_index in past_incorrect_ids:
                                error_count += 1
                                incorrect_ids.append(weight_index)

                            weights[weight_index] *= (error / (1.0 - error))
                base = sum(weights)
                for weight_index in range(0, len(weights)):
                    weights[weight_index] = weights[weight_index] / base
                stumps_left -= 1
                past_incorrect_ids = list(incorrect_ids)

    def test(self, test_row):
        votes = {}
        for orientation, stumps in self.stump_allocation.iteritems():
            orientation_acceptance = 0
            for stump in stumps:
                if test_row[stump[0]] > test_row[stump[1]]:
                    orientation_acceptance += 1
            votes[orientation] = (1.0 * orientation_acceptance) / len(stumps)
        return max(votes.iteritems(), key=operator.itemgetter(1))[0]

    def save(self, filename):
        with open(filename, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.stump_allocation.items():
                writer.writerow([key, value])

    def __str__(self):
        return pprint.pformat(self.stump_allocation)


class NNet(Model):
    def __init__(self):
        self.model_type = "nnet"
