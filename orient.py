# /usr/bin/env python

import sys
import csv
import pprint

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
train_rows = []
test_rows = []
csv.register_dialect(
    'space_dialect',
    delimiter=' ',
)
with open(train_file, "r") as train_file_handler:
    reader = csv.reader(train_file_handler, dialect="space_dialect")
    for row in reader:
        train_rows.append(row)

with open(test_file, "r") as test_file_handler:
    reader = csv.reader(test_file_handler, dialect="space_dialect")
    for row in reader:
        test_rows.append(row)
