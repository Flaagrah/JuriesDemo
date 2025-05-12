from jury_finetuning.run_jury import fine_tune_jury
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Positional arguments
    parser.add_argument("model_dir", help="Name of the model")
    parser.add_argument("model_nick_name", help="Name of the model")
    parser.add_argument("huggingface_secret", help="Secret to access model")

    args = parser.parse_args()

    # Get the model from the command line arguments
    model_dir = args.model_dir
    model_nick_name = args.model_nick_name
    
    os.environ['HF_TOKEN'] = args.huggingface_secret

    # Retrieve the datasets
    fine_tune_jury(model_dir, model_nick_name)


