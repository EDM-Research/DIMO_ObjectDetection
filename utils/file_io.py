import os.path
from typing import Tuple, List
from training.evaluation import ModelTest


def read_test_batch(file_path: str) -> List[ModelTest]:
    tests = []
    with open(file_path, 'r') as f:
        for line in f:
            line_split = line.split(",")
            assert len(line_split) == 2, "Batch test file should be in format 'model_id,subset'"

            model_id, subset = line_split
            tests.append(ModelTest(model_id, subset.rstrip()))
    return tests


def write_model_epochs(model_id: str, aps: list, epochs: list):
    filename = os.path.join('results', f"{model_id}.csv")
    with open(filename, 'w') as f:
        f.write("epoch,ap\n")
        for epoch, ap in zip(aps, epochs):
            f.write(f"{epoch},{ap}\n")


def write_test_metrics(tests: List[ModelTest], filename: str):

    with open(filename, 'w') as f:
        f.write("model_id,subset,ap,ap50,ap75\n")
        for test in tests:
            f.write(f"{test.model_id},{test.test_subset},{test.metrics['ap']},{test.metrics['ap_50']},{test.metrics['ap_75']}\n")
