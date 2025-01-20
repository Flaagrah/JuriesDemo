import ast
import pandas as pd
from utils import exact_match
import torch

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
        first_val = str(1-logits[(0 if is_correct else 1)]) if is_calib else str(1-logits[(0 if logits[0] > logits[1] else 1)])
        df.loc[index] = [first_val, is_correct == (logits[0] > logits[1]), str(bool(is_correct)), logits[0] > logits[1], max(logits)]

    df.to_csv("s_values_"+jury_model_name+".csv", index=False)   
    return file_name