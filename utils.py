import re
import string
import pandas as pd
import ast
from pandas import DataFrame
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from create_alpha_dict import create_q_alphas, get_s_vals
from find_epsilon_from_s import find_epsilon_from_s

BASE_DATA_FOLDER = "base_model_outputs/"

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, answers):
    """True if prediction matches any answer."""
    # Remove . and \n from prediction
    prediction = re.sub(r'[.,\n]', '', prediction)
    prediction = normalize_text(prediction)
    answers = [normalize_text(a) for a in answers]
    return float(any([prediction == a for a in answers]))

base_model = "jury_judgements"
def get_epsilon_dict(calib_file_name: str) -> dict[float, float]:
    file_name = generate_s_values(base_model, calib_file_name, True)
    return create_q_alphas(file_name)

def get_test_s_vals(test_file_name: str) -> DataFrame:
    file_name = generate_s_values(base_model, test_file_name, False)
    return get_s_vals(file_name)


def shuffle_jury_data(jury_model_name: str, jury_model_name_test: str, seed: int = 10):
    name = base_model+"/jury_logits_"+jury_model_name+".csv"
    test_name = base_model+"/jury_logits_"+jury_model_name_test+".csv"
    
    # Read the csv's as pandas dataframes
    df = pd.read_csv(name)
    df_test = pd.read_csv(test_name)
    # Combine the dataframes
    combined_df = pd.concat([df, df_test], ignore_index=True)
    
    # Shuffle the combined dataframe
    shuffled_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split back into original sizes
    df = shuffled_df.iloc[:len(df)]
    df_test = shuffled_df.iloc[len(df):]

    name_shuffled = base_model+"/jury_logits_"+jury_model_name+"_shuffled.csv"
    name_shuffled_test = base_model+"/jury_logits_"+jury_model_name_test+"_shuffled.csv"

    # Write the shuffled dataframes to new csv files
    df.to_csv(name_shuffled, index=False)
    df_test.to_csv(name_shuffled_test, index=False)

def generate_s_values(folder_name: str, jury_model_name: str, is_calib=True):
    file_name = "s_values_"+jury_model_name+".csv"
    df_jury_data = pd.read_csv(folder_name+"/jury_logits_"+jury_model_name+".csv")
    df = pd.DataFrame(columns=["s_val", "is_jury_accurate", "is_base_model_correct", "is_jury_approving", "max_logits"])
    for index, row in df_jury_data.iterrows():
        question = row['question']
        given_answer = row['answer']
        logits = ast.literal_eval(row['logits'])[0]
        correct_answers = ast.literal_eval(row['normalized_aliases'])
        # logits = ast.literal_eval(row['logits'].replace("tensor(", "").replace(")", ""))[0]
        #correct_answers = question_set[index]['answer']['normalized_aliases']
        
        is_correct = exact_match(given_answer, correct_answers)
        # The first val is the s_val (s_val of the true class for calibration and max class for test)
        # TODO is_jury_approving should be refactored into a separate function.
        first_val = str(1-logits[(0 if is_correct else 1)]) if is_calib else str(1-logits[(0 if logits[0] > logits[1] else 1)])
        df.loc[index] = [first_val, is_correct == (logits[0] > logits[1]), str(bool(is_correct)), logits[0] > logits[1], max(logits)]

    df.to_csv("s_values_"+jury_model_name+".csv", index=False)   
    return file_name

def calculate_metrics_for_response(juries_to_logits_epsilons):
    true_counts = 0
    false_counts = 0
    correct_score = 0
    incorrect_score = 0
    correct_mul_score = 1
    incorrect_mul_score = 1

    is_not_vetoed = True
    max_poll_logits = 0
    max_poll_logits_result = None
    max_poll_confidence = 0
    max_poll_confidence_result = None

    jury_info = {}
    for jury_name, logits_epsilons in juries_to_logits_epsilons.items():
        logits = logits_epsilons['logits']
        epsilon_to_s = logits_epsilons['epsilon_to_s']
        confidence = 1 - find_epsilon_from_s(min(logits), epsilon_to_s)
        # TODO refactor is_jury_approving to a separate function
        is_jury_approving = logits[0] > logits[1]
        if is_jury_approving:
            true_counts += 1
            correct_score += confidence - 0.5
            correct_mul_score *= (1 - confidence)
        else:
            false_counts += 1
            incorrect_score += confidence - 0.5
            incorrect_mul_score *= (1 - confidence)
            is_not_vetoed = False
        
        if max(logits) > max_poll_logits:
            max_poll_logits = max(logits)
            max_poll_logits_result = is_jury_approving
        if confidence > max_poll_confidence:
            max_poll_confidence = confidence
            max_poll_confidence_result = is_jury_approving
        
        jury_info[jury_name] = {
            "logits": logits,
            "confidence": confidence,
            "is_jury_approving": is_jury_approving
        }
        

    return {
        "Majority Vote": true_counts > false_counts,
        "Calibrated Multiplicative Score": correct_mul_score < incorrect_mul_score,
        "Calibrated Confidence Score": correct_score > incorrect_score,
        "Veto Poll": is_not_vetoed,
        "Max Poll (Logits)": max_poll_logits_result,
        "Max Poll (Confidence)": max_poll_confidence_result,
        "Jury Info": jury_info
    }

def get_quantized_model(model_dir: str, device) -> AutoModelForCausalLM:
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

    return model