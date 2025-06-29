
from find_epsilon_from_s import find_epsilon_from_s
from pandas import DataFrame
from utils import calculate_metrics_for_response
import pandas as pd

JURY1 = "llama13b"
JURY2 = "olmo13b"
JURY3 = "stable13b"

MAJORITY_VOTE = "Majority Vote"
CALIBRATED_CONFIDENCE_SCORE = "Calibrated Confidence Score"
CALIBRATED_MULTIPLICATIVE_SCORE = "Calibrated Multiplicative Score"
MAX_POLL_CONFIDENCE = "Max Poll (Confidence)"
MAX_POLL_LOGITS = "Max Poll (Logits)"
VETO_POLL = "Veto Poll"

EVAL_TRACK_LIST = [
    MAJORITY_VOTE,
    CALIBRATED_CONFIDENCE_SCORE,
    CALIBRATED_MULTIPLICATIVE_SCORE,
    MAX_POLL_CONFIDENCE,
    MAX_POLL_LOGITS,
    VETO_POLL
]

TRUE_POSITIVE = "True_Positive"
FALSE_NEGATIVE = "False_Negative"
FALSE_POSITIVE = "False_Positive"
TRUE_NEGATIVE = "True_Negative"
TOTAL_ACCURATE = "Total_Accurate"
ACCURACY_RATE = "Accuracy_Rate"
PRECISION = "Precision"
RECALL = "Recall"
F1_SCORE = "F1_Score"

METRIC_TRACK_LIST = [
    ACCURACY_RATE,
    PRECISION,
    RECALL,
    F1_SCORE
]

NUM_ACCURATE = "Num_Accurate"
NUM_ACCURATE_ON_DISAGREEMENTS = "Num_Accurate_On_Disagreements"
MAX_LOGITS = "Num_Max_Logits"
MAX_CONFIDENCE = "Num_Max_Confidence"

INTERVAL_START = 0.5
INTERVAL_END = 1.0

def get_eval_info_defaults() -> dict:
    """Returns a dictionary with default values for evaluation metrics."""
    return { TRUE_POSITIVE: 0, FALSE_NEGATIVE: 0, FALSE_POSITIVE: 0, TRUE_NEGATIVE: 0, TOTAL_ACCURATE: 0, PRECISION: 0, RECALL: 0, F1_SCORE: 0 }

def get_jury_eval_defaults() -> dict:
    """Returns a dictionary with default values for jury evaluation."""
    return {
        NUM_ACCURATE: 0,
        NUM_ACCURATE_ON_DISAGREEMENTS: 0,
        MAX_LOGITS: 0,
        MAX_CONFIDENCE: 0
    }

def get_eval_result(is_correct: bool, is_eval_approving: bool) -> str:
    """Determines the evaluation result based on correctness and approval status.
    :param is_correct: Boolean indicating if the base model's answer is correct.
    :param is_eval_approving: Boolean indicating if the jury evaluation approves the answer.
    :return: A string representing the evaluation result.
    """
    if is_correct and is_eval_approving:
        return TRUE_POSITIVE
    elif is_correct and not is_eval_approving:
        return FALSE_NEGATIVE
    elif not is_correct and is_eval_approving:
        return FALSE_POSITIVE
    else:
        return TRUE_NEGATIVE

# TODO: Refactor this function to use the calculate_metrics_for_response function
def generate_stats(jury1_test_vals: DataFrame, jury2_test_vals: DataFrame, jury3_test_vals: DataFrame, jury1_dict: dict, jury2_dict: dict, jury3_dict: dict):
    """    Print statistics about the jury evaluations.
    :param jury1_test_vals: DataFrame containing test values for jury 1.
    :param jury2_test_vals: DataFrame containing test values for jury 2.
    :param jury3_test_vals: DataFrame containing test values for jury 3.
    :param jury1_dict: Dictionary containing epsilon values for jury 1.
    :param jury2_dict: Dictionary containing epsilon values for jury 2.
    :param jury3_dict: Dictionary containing epsilon values for jury 3.
    """
    eval_info = {
        MAJORITY_VOTE: get_eval_info_defaults(),
        CALIBRATED_CONFIDENCE_SCORE: get_eval_info_defaults(),
        CALIBRATED_MULTIPLICATIVE_SCORE: get_eval_info_defaults(),
        MAX_POLL_CONFIDENCE: get_eval_info_defaults(),
        MAX_POLL_LOGITS: get_eval_info_defaults(),
        VETO_POLL: get_eval_info_defaults()
    }
    
    jury_info = {
        JURY1: get_jury_eval_defaults(),
        JURY2: get_jury_eval_defaults(),
        JURY3: get_jury_eval_defaults()
    }
    
    total_disagreements = 0
    base_model_correct = 0

    results_list = []
    len_points = len(jury1_test_vals)

    for i in range(len_points):
        
        j1_info = jury1_test_vals.loc[i]
        j2_info = jury2_test_vals.loc[i]
        j3_info = jury3_test_vals.loc[i]

        if j1_info['is_base_model_correct']:
            base_model_correct += 1
        
        j1_confidence = 1 - find_epsilon_from_s(j1_info['s_val'], jury1_dict)
        j2_confidence = 1 - find_epsilon_from_s(j2_info['s_val'], jury2_dict)
        j3_confidence = 1 - find_epsilon_from_s(j3_info['s_val'], jury3_dict)

        info_list = [{'name': JURY1, 'data': j1_info, 'confidence': j1_confidence, 's_val': j1_info['s_val']}, 
                     {'name': JURY2, 'data': j2_info, 'confidence': j2_confidence, 's_val': j2_info['s_val']}, 
                     {'name': JURY3, 'data': j3_info, 'confidence': j3_confidence, 's_val': j3_info['s_val']}]

        true_counts = 0
        false_counts = 0
        correct_score = 0
        incorrect_score = 0
        correct_mul_score = 1
        incorrect_mul_score = 1

        for index, information in enumerate(info_list):
            if information['data']['is_jury_approving']:
                true_counts += 1
                correct_score += information['confidence'] - 0.5
                correct_mul_score *= (1 - information['confidence'])
            else:
                false_counts += 1
                incorrect_score += information['confidence'] - 0.5
                incorrect_mul_score *= (1 - information['confidence'])

        def has_consensus():
            return true_counts == 0 or false_counts == 0

        # Loop through info_list
        for index, information in enumerate(info_list):
            if information['confidence'] >= INTERVAL_START and information['confidence'] <= INTERVAL_END:
                if information['data']['is_base_model_correct'] == information['data']['is_jury_approving']:
                    jury_info[information['name']][NUM_ACCURATE] += 1
                    if not has_consensus():
                        jury_info[information['name']][NUM_ACCURATE_ON_DISAGREEMENTS] += 1

        # Continue if they all agree
        if has_consensus():
            continue

        total_disagreements += 1

        # Count veto judgements
        eval_info[VETO_POLL][get_eval_result(j1_info['is_base_model_correct'], false_counts == 0)] += 1
    
        max_conf = [round(info_list[0]['confidence'] - 0.5, 2), round(info_list[1]['confidence'] - 0.5, 2), round(info_list[2]['confidence'] - 0.5, 2)]
        max_index_conf = max_conf.index(max(max_conf))
        jury_info[info_list[max_index_conf]['name']][MAX_CONFIDENCE] += 1
        # Max Confidence
        eval_info[MAX_POLL_CONFIDENCE][get_eval_result(info_list[max_index_conf]['data']['is_base_model_correct'], info_list[max_index_conf]['data']['is_jury_approving'])] += 1

        min_s_val = [round(info_list[0]['s_val'], 4), round(info_list[1]['s_val'], 4), round(info_list[2]['s_val'], 4)]
        # For each entry in sorted, get the index of the max entry
        max_logits_index = min_s_val.index(min(min_s_val))
        jury_info[info_list[max_logits_index]['name']][MAX_LOGITS] += 1
        # Max Logits
        eval_info[MAX_POLL_LOGITS][get_eval_result(info_list[max_logits_index]['data']['is_base_model_correct'], info_list[max_logits_index]['data']['is_jury_approving'])] += 1
        
        eval_info[MAJORITY_VOTE][get_eval_result(info_list[0]['data']['is_base_model_correct'], true_counts > false_counts)] += 1
        eval_info[CALIBRATED_CONFIDENCE_SCORE][get_eval_result(info_list[0]['data']['is_base_model_correct'], correct_score > incorrect_score)] += 1
        eval_info[CALIBRATED_MULTIPLICATIVE_SCORE][get_eval_result(info_list[0]['data']['is_base_model_correct'], correct_mul_score < incorrect_mul_score)] += 1

        # Create a row for the results dataframe
        row = {
            'llama_s_val': j1_info['s_val'],
            'olmo_s_val': j2_info['s_val'], 
            'stable_s_val': j3_info['s_val'],
            'llama_confidence': j1_confidence,
            'olmo_confidence': j2_confidence,
            'stable_confidence': j3_confidence,
            'lowest_s_val_model': ['llama', 'olmo', 'stable'][max_logits_index],
            'lowest_s_val_model_answer': info_list[max_logits_index]['data']['is_jury_approving'],
            'highest_confidence_model': ['llama', 'olmo', 'stable'][max_index_conf],
            'highest_confidence_model_answer': info_list[max_index_conf]['data']['is_jury_approving'],
            'is_base_model_correct': j1_info['is_base_model_correct'],
        }

        # Add the row to the results dataframe
        results_list.append(row)
        #results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results.csv")

    for key, value in eval_info.items():
        eval_info[key]["Correct"] = value[TRUE_POSITIVE] + value[TRUE_NEGATIVE]

    print("Accurate vs Inaccurate judgements")
    print("Majority Poll:", eval_info[MAJORITY_VOTE]["Correct"], total_disagreements - eval_info[MAJORITY_VOTE]["Correct"], eval_info[MAJORITY_VOTE]["Correct"]/(total_disagreements))
    print("Calibrated Confidence Poll:", eval_info[CALIBRATED_CONFIDENCE_SCORE]["Correct"], total_disagreements - eval_info[CALIBRATED_CONFIDENCE_SCORE]["Correct"], eval_info[CALIBRATED_CONFIDENCE_SCORE]["Correct"]/(total_disagreements))
    print("Calibrated Mul Confidence Poll:", eval_info[CALIBRATED_MULTIPLICATIVE_SCORE]["Correct"], total_disagreements - eval_info[CALIBRATED_MULTIPLICATIVE_SCORE]["Correct"], eval_info[CALIBRATED_MULTIPLICATIVE_SCORE]["Correct"]/(total_disagreements))
    print("Calibrated Max Poll:", eval_info[MAX_POLL_CONFIDENCE]["Correct"], total_disagreements - eval_info[MAX_POLL_CONFIDENCE]["Correct"], eval_info[MAX_POLL_CONFIDENCE]["Correct"]/(total_disagreements))
    print("Max Poll (Uncalibrated):", eval_info[MAX_POLL_LOGITS]["Correct"], total_disagreements - eval_info[MAX_POLL_LOGITS]["Correct"], eval_info[MAX_POLL_LOGITS]["Correct"]/(total_disagreements))
    print("Veto Poll:", eval_info[VETO_POLL]["Correct"], total_disagreements - eval_info[VETO_POLL]["Correct"], eval_info[VETO_POLL]["Correct"]/(total_disagreements))
    print("Majorities:", eval_info[MAJORITY_VOTE])
    print("Calibrated Confidence Score:", eval_info[CALIBRATED_CONFIDENCE_SCORE])
    print("Calibrated Multiplicative Score:", eval_info[CALIBRATED_MULTIPLICATIVE_SCORE])
    print("Max Poll (Confidence):", eval_info[MAX_POLL_CONFIDENCE])
    print("Max Poll (Logits):", eval_info[MAX_POLL_LOGITS])
    print("Veto Poll:", eval_info[VETO_POLL])
    print("Start Interval:", INTERVAL_START)
    print("End Interval:", INTERVAL_END)
    print(jury_info)
    # Put all of the printed information into a dictionary
    eval_results = {
        "eval_info": eval_info,
        "jury_info": jury_info,
        "total_disagreements": total_disagreements,
        "base_model_correct": base_model_correct,
    }
    # Print precision and recall for each evaluation metric
    for key, value in eval_info.items():
        if value[TRUE_POSITIVE] + value[FALSE_POSITIVE] > 0:
            precision = value[TRUE_POSITIVE] / (value[TRUE_POSITIVE] + value[FALSE_POSITIVE])
        else:
            precision = 0
        if value[TRUE_POSITIVE] + value[FALSE_NEGATIVE] > 0:
            recall = value[TRUE_POSITIVE] / (value[TRUE_POSITIVE] + value[FALSE_NEGATIVE])
        else:
            recall = 0
        # Print F1 scores
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{key} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
        eval_info[key][PRECISION] = precision
        eval_info[key][RECALL] = recall
        eval_info[key][F1_SCORE] = f1_score
        eval_info[key][TOTAL_ACCURATE] = value[TRUE_POSITIVE] + value[TRUE_NEGATIVE]
        eval_info[key][ACCURACY_RATE] = (value[TRUE_POSITIVE] + value[TRUE_NEGATIVE]) / (total_disagreements if total_disagreements > 0 else 1)

    return eval_results

def generate_stats2(llama_test_vals: DataFrame, olmo_test_vals: DataFrame, stable_test_vals: DataFrame, llama_dict: dict, olmo_dict: dict, stable_dict: dict):
    def get_logits(row):
        if not row['is_jury_approving']:
            return [row['s_val'], 1 - row['s_val']]
        else :
            return [1 - row['s_val'], row['s_val']]

    majority_vote = 0
    cal_confidence_poll = 0
    cal_mul_confidence_poll = 0
    cal_max_poll = 0
    max_poll_uncalibrated = 0
    veto_poll = 0
    total = 0
    for i in range(len(llama_test_vals)):
        # Get ith row of each DataFrame
        ll_info = llama_test_vals.iloc[i]
        ol_info = olmo_test_vals.iloc[i]
        st_info = stable_test_vals.iloc[i]

        if ll_info['is_jury_approving'] == ol_info['is_jury_approving'] == st_info['is_jury_approving']:
            continue

        total += 1
        ll_logits = get_logits(ll_info)
        ol_logits = get_logits(ol_info)
        st_logits = get_logits(st_info)

        result = calculate_metrics_for_response({
            'llama': {'logits': ll_logits, 'epsilon_to_s': llama_dict},
            'olmo': {'logits': ol_logits, 'epsilon_to_s': olmo_dict},
            'stable': {'logits': st_logits, 'epsilon_to_s': stable_dict}
        })

        is_correct = ll_info['is_base_model_correct']
        if result['Majority Vote'] == is_correct:
            majority_vote += 1
        if result['Calibrated Confidence Score'] == is_correct:
            cal_confidence_poll += 1
        if result['Calibrated Multiplicative Score'] == is_correct:
            cal_mul_confidence_poll += 1
        if result['Max Poll (Confidence)'] == is_correct:
            cal_max_poll += 1
        if result['Max Poll (Logits)'] == is_correct:
            max_poll_uncalibrated += 1
        if result['Veto Poll'] == is_correct:
            veto_poll += 1

    print("Majority Vote:", majority_vote, total - majority_vote, majority_vote / total)
    print("Calibrated Confidence Poll:", cal_confidence_poll, total - cal_confidence_poll, cal_confidence_poll / total)
    print("Calibrated Multiplicative Poll:", cal_mul_confidence_poll, total - cal_mul_confidence_poll, cal_mul_confidence_poll / total)
    print("Calibrated Max Poll:", cal_max_poll, total - cal_max_poll, cal_max_poll / total)
    print("Max Poll (Uncalibrated):", max_poll_uncalibrated, total - max_poll_uncalibrated, max_poll_uncalibrated / total)
    print("Veto Poll:", veto_poll, total - veto_poll, veto_poll / total)

def get_paths() -> list[list]:
    """Returns a list of paths to the statistics to be aggregated.
    :return: A list of lists, where each inner list contains the keys to access the statistics.
    """
    list_of_paths = [[adj, metric] for adj in EVAL_TRACK_LIST for metric in METRIC_TRACK_LIST]
    return list_of_paths

def aggregate_stats(stats_list: list[dict]) -> dict:
    """Aggregate statistics from multiple runs.
    :param stats_list: List of dictionaries containing statistics from different runs.
    :param paths: List of paths to the files containing the statistics.
    :return: A dictionary with aggregated statistics.
    """
    paths = get_paths()
    aggregated = {}
    
    # Iterate through each stats in stats_list and aggregate the values
    for stats in stats_list:
        for path in paths:
            current = aggregated
            current_stats = stats['eval_info']
            # print(f"Current stats: {current_stats}")
            # print(".............................................")
            for i, key in enumerate(path):
                # print(f"Processing key: {key}, Index: {i}, Current stats: {current_stats}")
                if i == len(path) - 1:
                    if key not in current:
                        current[key] = 0
                    current[key] += current_stats[key] / len(stats_list)
                else:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                    # print(f"Current path: {path}, Current key: {current}, Current stats: {current_stats}")
                    current_stats = current_stats[key]
    return aggregated
