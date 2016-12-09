# /usr/bin/env python

import sys
import csv
import itertools
from Model1 import Nearest, AdaBoost, NNet

train_file = "train_file.txt"
test_file = "test_file.txt"
method = ""
parameter = None
model_file = "best.model"
metadata_file = "metadata.csv"
try:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    method = sys.argv[3]
except IndexError:
    print("Usage: python orient1.py train_file test_file algorithm [model-parameter]")
    sys.exit(1)
if method in ["adaboost", "nnet"]:
    try:
        parameter = sys.argv[4]
    except IndexError:
        print("Usage: python orient1.py train_file test_file algorithm model-parameter")
        print("model-parameter (stump_count for adaboost and hidden_count for nnet) is required")
        sys.exit(2)
if method in ["best"]:
    try:
        model_file = sys.argv[4]
    except IndexError:
        print("You have chosen to not supply model file. We will use file with name \"best.model\" if it exists. If "
              "not we use metadata to train and test using best model")
header = ["id", "orientation"]
color_indices = [color + item for item in
                 [str(i[0]) + str(i[1]) for i in itertools.product([1, 2, 3, 4, 5, 6, 7, 8], repeat=2)] for color in
                 ['r', 'g', 'b']]
indices = header + color_indices
select_indices = [color + item for item in
                  [str(i[0]) + str(i[1]) for i in itertools.product([1, 2, 3, 4, 5, 6, 7, 8], repeat=2) if
                   i[0] == 1 or i[1] == 1 or i[0] == 8 or i[1] == 8] for color in ['r', 'g']]
# for color in ['r', 'g', 'b']:
#     for corner in 1, 8:
#         for other in [1, 2, 3, 4, 5, 6, 7, 8]:
#             select_indices.append(color + str(corner) + str(other))


train_rows = []
test_rows = []
csv.register_dialect(
    'space_dialect',
    delimiter=' ',
)
with open(train_file, "r") as train_file_handler:
    reader = csv.reader(train_file_handler, dialect="space_dialect")
    for row in reader:
        current_row = []
        for column in row:
            try:
                current_row.append(int(column))
            except ValueError:
                current_row.append(column)
        current_dict = dict(zip(indices, current_row))
        train_rows.append(current_dict)

with open(test_file, "r") as test_file_handler:
    reader = csv.reader(test_file_handler, dialect="space_dialect")
    for row in reader:
        current_row = []
        for column in row:
            try:
                current_row.append(int(column))
            except ValueError:
                current_row.append(column)
        current_dict = dict(zip(indices, current_row))
        test_rows.append(current_dict)
print("Data set ready")
model = None
if method == "nearest":
    model = Nearest(color_indices)
    for train_item in train_rows:
        model.train(train_item)
elif method == "adaboost":
    model = AdaBoost(color_indices, int(parameter))
    model.train(train_rows)
    print(model)

elif method == "nnet":
    model = NNet(int(parameter))
    for train_item in train_rows:
        model.train(train_item)
        # break


successes = 0
exit(0)
totals = 0
print("Training complete")
confusion_matrix = {"0": {"0": 0, "90": 0, "180": 0, "270": 0}, "90": {"0": 0, "90": 0, "180": 0, "270": 0},
                    "180": {"0": 0, "90": 0, "180": 0, "270": 0}, "270": {"0": 0, "90": 0, "180": 0, "270": 0}}
for test_item in test_rows:
    totals += 1
    id, orientation = model.test(test_item)
    if orientation == test_item["orientation"]:
        successes += 1
    confusion_matrix[str(test_item["orientation"])][str(orientation)] += 1
print("Confusion Matrix")
print("\t0\t90\t180\t270\t")
# for key in confusion_matrix.iterkeys():
print("0\t" + str(confusion_matrix["0"]["0"]) + "\t" + str(confusion_matrix["0"]["90"]) + "\t" + str(
    confusion_matrix["0"]["180"]) + "\t" + str(confusion_matrix["0"]["270"]))
print("90\t" + str(confusion_matrix["90"]["0"]) + "\t" + str(confusion_matrix["90"]["90"]) + "\t" + str(
    confusion_matrix["90"]["180"]) + "\t" + str(confusion_matrix["90"]["270"]))
print("180\t" + str(confusion_matrix["180"]["0"]) + "\t" + str(confusion_matrix["180"]["90"]) + "\t" + str(
    confusion_matrix["180"]["180"]) + "\t" + str(confusion_matrix["180"]["270"]))
print("270\t" + str(confusion_matrix["270"]["0"]) + "\t" + str(confusion_matrix["270"]["90"]) + "\t" + str(
    confusion_matrix["270"]["180"]) + "\t" + str(confusion_matrix["270"]["270"]))
print(successes)
print(totals)
print(1.0 * successes / totals)
