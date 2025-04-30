from base_model_data_creation.run_base_model import run_base_model
from data_retrieval.get_data_sets import retrieve_data_sets
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Positional arguments
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument("huggingface_secret", help="Secret to access model")

    args = parser.parse_args()

    # Get the model name from the command line arguments
    model_name = args.model_name
    
    os.environ['HF_TOKEN'] = args.huggingface_secret

    # Retrieve the datasets
    fine_tune_data, fine_tune_test_data, calibration_data, test_data = retrieve_data_sets()
    run_base_model(model_name, fine_tune_data, fine_tune_test_data, calibration_data, test_data)


