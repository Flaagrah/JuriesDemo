
from find_epsilon_from_s import find_epsilon_from_s
from pandas import DataFrame
import pandas as pd

def print_stats(llama_test_vals: DataFrame, olmo_test_vals: DataFrame, stable_test_vals: DataFrame, llama_dict: dict, olmo_dict: dict, stable_dict: dict):
    
    total_disagreements = 0
    
    # majority poll vs calibrated poll
    accurate_judgements = 0
    cal_acc_judgements = 0
    cal_mul_acc_judgements = 0
    max_cal_acc_judgements = 0
    max_acc_judgements = 0
    veto_acc_judgements = 0

    true_majority = 0
    false_majority = 0
    true_majority_true = 0
    true_majority_false = 0
    false_majority_true = 0
    false_majority_false = 0


    llama_pd_data = []
    olmo_pd_data = []
    stable_pd_data = []

    interval_start = 0.5
    interval_end = 1

    jury_correct = [0, 0, 0]
    jury_correct_disagreements = [0, 0, 0]
    max_logits = [0, 0, 0]
    max_confidence = [0, 0, 0]

    base_model_correct = 0

    # Initialize DataFrame with specified columns
    # results_df = pd.DataFrame(columns=[
    #     'llama_s_val', 
    #     'olmo_s_val',
    #     'stable_s_val',
    #     'llama_conf', 
    #     'olmo_conf',
    #     'stable_conf',
    #     'max_logits_judgement',
    #     'max_conf_judgement',
    #     'correct_judgement'
    # ])
    results_list = []
    len_points = len(llama_test_vals)
    #len_points = 50

    for i in range(len_points):
        
        ll_info = llama_test_vals.loc[i]
        ol_info = olmo_test_vals.loc[i]
        st_info = stable_test_vals.loc[i]

        if ll_info['is_base_model_correct']:
            base_model_correct += 1
        
        llama_confidence = 1 - find_epsilon_from_s(ll_info['s_val'], llama_dict)
        olmo_confidence = 1 - find_epsilon_from_s(ol_info['s_val'], olmo_dict)
        stable_confidence = 1 - find_epsilon_from_s(st_info['s_val'], stable_dict)

        info_list = [{'data': ll_info, 'confidence': llama_confidence}, 
                     {'data': ol_info, 'confidence': olmo_confidence}, 
                     {'data': st_info, 'confidence': stable_confidence}]

        llama_pd_data.append([ll_info['s_val'], llama_confidence, ll_info['is_base_model_correct'] == ll_info['is_jury_approving'], ll_info['is_jury_approving']])
        olmo_pd_data.append([ol_info['s_val'], olmo_confidence, ol_info['is_base_model_correct'] == ol_info['is_jury_approving'], ol_info['is_jury_approving']])
        stable_pd_data.append([st_info['s_val'], stable_confidence, st_info['is_base_model_correct'] == st_info['is_jury_approving'], st_info['is_jury_approving']])

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
            if information['confidence'] >= interval_start and information['confidence'] <= interval_end:
                if information['data']['is_base_model_correct'] == information['data']['is_jury_approving']:
                    jury_correct[index] += 1
                    if not has_consensus():
                        jury_correct_disagreements[index] += 1

        # Continue if they all agree
        if has_consensus():
            continue

        total_disagreements += 1

        # Count veto judgements
        if false_counts > 0:
            if not ll_info['is_base_model_correct']:
                veto_acc_judgements += 1

        # Count majorities
        if true_counts > false_counts:
            true_majority += 1
            if ll_info['is_base_model_correct']:
                true_majority_true += 1
            else:
                true_majority_false += 1
        else: 
            false_majority += 1
            if ll_info['is_base_model_correct']:
                false_majority_true += 1
            else:
                false_majority_false += 1
        conf_sorted = [round(llama_confidence - 0.5, 2), round(olmo_confidence - 0.5, 2), round(stable_confidence - 0.5, 2)]
        #print(sorted)
        # For each entry in sorted, get the index of the max entry
        max_index_conf = conf_sorted.index(max(conf_sorted))
        max_confidence[max_index_conf] += 1
        # Max Confidence
        if info_list[max_index_conf]['data']['is_base_model_correct'] == info_list[max_index_conf]['data']['is_jury_approving']:
            max_cal_acc_judgements += 1

        max_sorted = [round(ll_info['s_val'], 4), round(ol_info['s_val'], 4), round(st_info['s_val'], 4)]
        #print(sorted)
        # For each entry in sorted, get the index of the max entry
        max_index = max_sorted.index(min(max_sorted))
        max_logits[max_index] += 1
        # Max Logits
        if info_list[max_index]['data']['is_base_model_correct'] == info_list[max_index]['data']['is_jury_approving']:
            max_acc_judgements += 1

        def model_correct_poll(condition):
            if condition:
                if ll_info['is_base_model_correct']:
                    return 1
            elif not ll_info['is_base_model_correct']:
                    return 1
            return 0
        
        accurate_judgements += model_correct_poll(true_counts > false_counts)
        cal_acc_judgements += model_correct_poll(correct_score > incorrect_score)
        cal_mul_acc_judgements += model_correct_poll(correct_mul_score < incorrect_mul_score)

        # Create a row for the results dataframe
        row = {
            'llama_s_val': ll_info['s_val'],
            'olmo_s_val': ol_info['s_val'], 
            'stable_s_val': st_info['s_val'],
            'llama_confidence': llama_confidence,
            'olmo_confidence': olmo_confidence,
            'stable_confidence': stable_confidence,
            'lowest_s_val_model': ['llama', 'olmo', 'stable'][max_index],
            'lowest_s_val_model_answer': info_list[max_index]['data']['is_jury_approving'],
            'highest_confidence_model': ['llama', 'olmo', 'stable'][max_index_conf],
            'highest_confidence_model_answer': info_list[max_index_conf]['data']['is_jury_approving'],
            'is_base_model_correct': ll_info['is_base_model_correct'],
        }

        # Add the row to the results dataframe
        results_list.append(row)
        #results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("results.csv")
    print("Accurate vs Inaccurate judgements")
    print("Majority Poll:", accurate_judgements, total_disagreements - accurate_judgements, accurate_judgements/(total_disagreements))
    print("Calibrated Confidence Poll:", cal_acc_judgements, total_disagreements - cal_acc_judgements, cal_acc_judgements/(total_disagreements))
    print("Calibrated Mul Confidence Poll:", cal_mul_acc_judgements, total_disagreements - cal_mul_acc_judgements, cal_mul_acc_judgements/(total_disagreements))
    print("Calibrated Max Poll:", max_cal_acc_judgements, total_disagreements - max_cal_acc_judgements, max_cal_acc_judgements/(total_disagreements))
    print("Max Poll (Uncalibrated):", max_acc_judgements, total_disagreements - max_acc_judgements, max_acc_judgements/(total_disagreements))
    print("Veto Poll:", veto_acc_judgements, total_disagreements - veto_acc_judgements, veto_acc_judgements/(total_disagreements))
    print("Majorities:", true_majority, false_majority)
    print("True Majority:", true_majority_true, true_majority_false)
    print("False Majority:", false_majority_false, false_majority_true)
    print("Start Interval:", interval_start)
    print("End Interval:", interval_end)
    print("Llama correct:", jury_correct[0], len_points - jury_correct[0], jury_correct[0]/len_points)
    print("Olmo correct:", jury_correct[1], len_points - jury_correct[1], jury_correct[1]/len_points)
    print("Stable correct:", jury_correct[2], len_points - jury_correct[2], jury_correct[2]/len_points)
    print("Base Model Correct:", base_model_correct, len_points - base_model_correct)
    print("Max Logits:", max_logits)
    print("Max Confidence:", max_confidence)
    print("Jury Correct on disagreements:", jury_correct_disagreements)
    print("Jury Correct on disagreements (normalized):", [x/total_disagreements for x in jury_correct_disagreements])

    return [accurate_judgements/(total_disagreements), cal_acc_judgements/(total_disagreements), cal_mul_acc_judgements/(total_disagreements), max_cal_acc_judgements/(total_disagreements), max_acc_judgements/(total_disagreements)]