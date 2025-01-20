# import the dateutil library
from datasets import load_dataset
from create_s_files import generate_s_values
from create_alpha_dict import create_q_alphas, get_s_vals
from print_stats import print_stats
from pandas import DataFrame
import pandas as pd

base_model = "cohere7b"
def get_epsilon_dict(calib_file_name: str) -> dict[float, float]:
    file_name = generate_s_values(base_model, calib_file_name, True)
    return create_q_alphas(file_name)

def get_test_s_vals(test_file_name: str) -> DataFrame:
    file_name = generate_s_values(base_model, test_file_name, False)
    return get_s_vals(file_name)


def shuffle_jury_data(jury_model_name: str, jury_model_name_test: str):
    name = base_model+"/jury_logits_"+jury_model_name+".csv"
    test_name = base_model+"/jury_logits_"+jury_model_name_test+".csv"
    
    # Read the csv's as pandas dataframes
    df = pd.read_csv(name)
    df_test = pd.read_csv(test_name)
    # Combine the dataframes
    combined_df = pd.concat([df, df_test], ignore_index=True)
    
    # Shuffle the combined dataframe
    shuffled_df = combined_df.sample(frac=1, random_state=32).reset_index(drop=True)
    
    # Split back into original sizes
    df = shuffled_df.iloc[:len(df)]
    df_test = shuffled_df.iloc[len(df):]

    name_shuffled = base_model+"/jury_logits_"+jury_model_name+"_shuffled.csv"
    name_shuffled_test = base_model+"/jury_logits_"+jury_model_name_test+"_shuffled.csv"

    # Write the shuffled dataframes to new csv files
    df.to_csv(name_shuffled, index=False)
    df_test.to_csv(name_shuffled_test, index=False)
    
shuffle_jury_data("olmo13b", "olmo13b_test")
shuffle_jury_data("llama13b", "llama13b_test")
shuffle_jury_data("stable13b", "stable13b_test")

# olmo13b_alphas = get_epsilon_dict("olmo13b")
# llama13b_alphas = get_epsilon_dict("llama13b")
# stable13b_alphas = get_epsilon_dict("stable13b")

# olmo13b_test = get_test_s_vals("olmo13b_test")
# llama13b_test = get_test_s_vals("llama13b_test")
# stable13b_test = get_test_s_vals("stable13b_test")

olmo13b_alphas = get_epsilon_dict("olmo13b_shuffled")
llama13b_alphas = get_epsilon_dict("llama13b_shuffled")
stable13b_alphas = get_epsilon_dict("stable13b_shuffled")

olmo13b_test = get_test_s_vals("olmo13b_test_shuffled")
llama13b_test = get_test_s_vals("llama13b_test_shuffled")
stable13b_test = get_test_s_vals("stable13b_test_shuffled")
'''
Accurate vs Inaccurate judgements
Majority Poll: 291 254
Calibrated Confidence Poll: 297 248
Calibrated Mul Confidence Poll: 321 224
Calibrated Max Poll: 367 178
Max Poll (Uncalibrated): 205 340
Veto Poll: 208 337
Majorities: 161 384
True Majority: 122 39
False Majority: 215 169
Start Interval: 0.5
End Interval: 1
Llama correct: 655 345 1.201834862385321
Olmo correct: 687 313 1.2605504587155962
Stable correct: 799 201 1.4660550458715595
Base Model Correct: 368 632
'''

print_stats(llama13b_test, olmo13b_test, stable13b_test, llama13b_alphas, olmo13b_alphas, stable13b_alphas)

# df_calib = pd.read_csv('s_values_llama13b.csv')
# df_test = pd.read_csv('s_values_llama13b_test.csv')
# print(df_calib.head(10))
# print(df_test.head(10))
# df_calib = df_calib.loc[0:10, :]
# df_test = df_test.loc[0:10, :]
# calib_correct = 0
# calib_incorrect = 0
# test_correct = 0
# test_incorrect = 0
# for i in range(len(df_calib)):
#     if df_calib.loc[i, 'is_base_model_correct']:
#         calib_correct += 1
#     else:
#         calib_incorrect += 1
#     if df_test.loc[i, 'is_base_model_correct']:
#         test_correct += 1
#     else:
#         test_incorrect += 1
# print(calib_correct, calib_incorrect)
# print(test_correct, test_incorrect)