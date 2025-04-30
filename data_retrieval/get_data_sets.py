from datasets import load_dataset

FINE_TUNE_SET_SIZE = 10000
FINE_TUNE_TEST_SET_SIZE = 1000
CALIBRATION_SET_SIZE = 1000
TEST_SET_SIZE = 1000

def retrieve_data_sets():

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

