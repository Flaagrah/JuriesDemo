from jury_finetuning.judge_response import call_jury
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import BASE_DATA_FOLDER
import os
import torch
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Positional arguments
    parser.add_argument("model_dir", help="Name of the model")
    parser.add_argument("model_name", help="Name of the model")
    parser.add_argument("huggingface_secret", help="Secret to access model")

    args = parser.parse_args()

    # Get the model name from the command line arguments
    model_dir = args.model_dir
    model_name = args.model_name
    os.environ['HF_TOKEN'] = args.huggingface_secret

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Set up 4-bit quantization configuration via bitsandbytes.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat 4 (NF4) quantization
        bnb_4bit_use_double_quant=True,       # Use double quantization for improved accuracy
        bnb_4bit_compute_dtype=torch.bfloat16 # Use BF16 for compute
    )

    # # Load the model with 4-bit quantization.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto"
    ).to(device)

    call_jury(model, tokenizer, "qlora_"+model_name+"_finetuned", BASE_DATA_FOLDER+"fine_tune_test_data_correctness.csv")
    call_jury(model, tokenizer, "qlora_"+model_name+"_calib", BASE_DATA_FOLDER+"calibration_data_correctness.csv")
    call_jury(model, tokenizer, "qlora_"+model_name+"_test", BASE_DATA_FOLDER+"test_data_correctness.csv")


