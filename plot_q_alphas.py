import math
import matplotlib.pyplot as plt
import numpy as np

s_vals_paper = []
s_vals_info = []
with open("s_values_stable13b_test.txt", "r") as f:
    lines = f.readlines()
    for i in lines:
        strs = i.split()
        s_vals_paper.append(float(strs[0]))
        s_vals_info.append((float(strs[0]), float(strs[4]), strs[1] == 'True'))

alphas = np.linspace(0.01, 0.99, 99)
q_alphas = []
for alpha in alphas:
    which_quantile = np.ceil((1 - alpha)*(len(s_vals_paper) + 1))/len(s_vals_paper)
    q_alpha = np.quantile(s_vals_paper, which_quantile, method='higher')
    #q_alpha = np.quantile(s_vals_paper, 1-alpha)
    q_alphas.append(q_alpha)
s_vals_paper.sort(reverse=True)
s_vals_dec = []
for i in range(10, len(s_vals_paper), 10):
    s_vals_dec.append(s_vals_paper[i])

plt.plot(alphas, q_alphas)