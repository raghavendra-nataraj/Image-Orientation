# /usr/bin/env python

import sys

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
except KeyError:
    print("Usage: python orient.py train_file test_file algorithm [model-parameter]")
    sys.exit(1)
if method in ["adaboost", "nnet"]:
    try:
        parameter = sys.argv[4]
    except KeyError:
        print("Usage: python orient.py train_file test_file algorithm model-parameter")
        print("model-parameter (stump_count for adaboost and hidden_count for nnet) is required")
        sys.exit(2)
if method in ["best"]:
    try:
        model_file = sys.argv[4]
    except KeyError:
        print("You have chosen to not supply model file. We will use file with name \"best.model\" if it exists. If "
              "not we use metadata to train and test using best model")
