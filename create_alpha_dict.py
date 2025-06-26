import numpy as np
import pandas as pd
from pandas import DataFrame


def create_alpha_dict(alphas: list[float], q_alphas: list[float]) -> dict[float, float]:
    """
    Create a dictionary mapping alpha values to their corresponding quantiles.
    :param alphas: A list of alpha values.
    :param q_alphas: A list of quantiles corresponding to the alpha values.
    :return: A dictionary where keys are alpha values and values are the corresponding quantiles.
    """
    alpha_dict = {}
    for i, alpha in enumerate(alphas):
        alpha_dict[round(alpha, 2)] = q_alphas[i]
    return alpha_dict

def create_q_alphas(file_name: str) -> dict[float, float]:
    """
    Create a dictionary of quantiles for each alpha value from the s_vals in the given CSV file.
    :param file_name: The name of the CSV file containing s_vals.
    :return: A dictionary where keys are alpha values and values are the corresponding quantiles of s_vals.
    """
    # Read the file into a pandas dataframe
    df = pd.read_csv(file_name)
    # Read the values in the s_vals column of df into the s_vals_paper list
    s_vals = df.loc[:, 's_val'].tolist()
    alphas = np.linspace(0.01, 0.99, 99)

    q_alphas = []
    for alpha in alphas:
        which_quantile = np.ceil((1 - alpha)*(len(s_vals) + 1))/len(s_vals)
        q_alpha = np.quantile(s_vals, which_quantile, method='higher')
        q_alphas.append(q_alpha)
    
    s_vals.sort(reverse=True)
    return create_alpha_dict(alphas, q_alphas)

def get_s_vals(file_name: str) -> DataFrame:
    """
    Get the s_vals from a CSV file.
    :param file_name: The name of the CSV file containing s_vals.
    :return: A DataFrame containing the s_vals, is_base_model_correct, and is_jury_approving columns.
    """
    df = pd.read_csv(file_name)
    return df[['s_val', 'is_base_model_correct', 'is_jury_approving']]
