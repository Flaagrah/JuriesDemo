from datasets import load_dataset
from datasets.dataset_dict import Dataset

FINE_TUNE_SET_SIZE = 5000
FINE_TUNE_TEST_SET_SIZE = 1000
CALIBRATION_SET_SIZE = 1000
TEST_SET_SIZE = 1000

def retrieve_data_sets() -> tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Retrieve the datasets for fine-tuning, testing, calibration, and final testing.
    Returns:
        tuple: A tuple containing the datasets for fine-tuning, fine-tune testing, calibration, and final testing.
    """
    dataset = load_dataset('trivia_qa', 'rc')

    dataset_train = dataset['train']
    # Shuffle the dataset
    dataset_shuffled = dataset_train.shuffle(seed=42)

    # Partition the dataset into different parts
    dataset_fine_tune = dataset_shuffled.select(range(FINE_TUNE_SET_SIZE))
    dataset_fine_tune_test = dataset_shuffled.select(range(FINE_TUNE_SET_SIZE, FINE_TUNE_SET_SIZE + FINE_TUNE_TEST_SET_SIZE))
    dataset_calib = dataset_shuffled.select(range(FINE_TUNE_SET_SIZE + FINE_TUNE_TEST_SET_SIZE, FINE_TUNE_SET_SIZE + FINE_TUNE_TEST_SET_SIZE + CALIBRATION_SET_SIZE))
    dataset_test = dataset_shuffled.select(range(FINE_TUNE_SET_SIZE + FINE_TUNE_TEST_SET_SIZE + CALIBRATION_SET_SIZE, FINE_TUNE_SET_SIZE + FINE_TUNE_TEST_SET_SIZE + CALIBRATION_SET_SIZE + TEST_SET_SIZE))

    return dataset_fine_tune, dataset_fine_tune_test, dataset_calib, dataset_test

