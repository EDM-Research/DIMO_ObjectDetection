from typing import Tuple, List


def read_test_batch(file_path: str) -> List[List[str]]:
    tests = []
    with open(file_path, 'r') as f:
        for line in f:
            line_split = line.split(",")
            assert len(line_split) == 2, "Batch test file should be in format 'model_id,subset'"

            model_id, subset = line_split
            tests.append([model_id, subset.rstrip()])
    return tests
