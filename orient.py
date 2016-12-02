# /usr/bin/env python

import sys
import csv
import itertools
from Model import Nearest

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
    print("Usage: python orient.py train_file test_file algorithm [model-parameter]")
    sys.exit(1)
if method in ["adaboost", "nnet"]:
    try:
        parameter = sys.argv[4]
    except IndexError:
        print("Usage: python orient.py train_file test_file algorithm model-parameter")
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
model = None
if method == "nearest":
    model = Nearest(color_indices)

for train_item in train_rows:
    model.train(train_item)
successes = 0
totals = 0
for test_item in test_rows:
    totals += 1
    id, orientation = model.test(test_item)
    if orientation == test_item["orientation"]:
        successes += 1
print(1.0 * successes / totals)
