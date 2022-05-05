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


def write_test_metrics(tests: List[ModelTest], filename: str):

    with open(filename, 'w') as f:
        f.write("model_id,subset,map,precision,recall\n")
        for test in tests:
            f.write(f"{test.model_id},{test.test_subset},{test.metrics['map']},{test.metrics['precision']},{test.metrics['recall']}\n")
