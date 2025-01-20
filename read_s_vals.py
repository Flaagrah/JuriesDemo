m_name = "stable13b_test"

def read_s_vals(model_name: str):
    with open("s_values_"+model_name+".txt", "r") as f:
        lines = f.readlines()
        s_vals = []
        for i in lines:
            strs = i.split()
            s_vals.append((float(strs[0]), float(strs[4]), strs[1] == 'True', strs[3] == 'True', strs[2] == 'True'))
        s_vals.sort(reverse=True)
        f.close()
    return s_vals

s_vals = read_s_vals(m_name)

f_count = 0
t_count = 0
j_correct_m_correct_count = 0
j_correct_m_incorrect_count = 0
j_incorrect_m_correct_count = 0
j_incorrect_m_incorrect_count = 0
for s in s_vals:
    if s[2]:
        t_count += 1
    else:
        f_count += 1
    if s[3] and s[4]:
        j_correct_m_correct_count += 1
    elif s[3] and not s[4]:
        j_correct_m_incorrect_count += 1
    elif not s[3] and s[4]:
        j_incorrect_m_correct_count += 1
    else:
        j_incorrect_m_incorrect_count += 1

print(t_count)
print(f_count)
print("j_correct, m_correct", j_correct_m_correct_count)
print("j_correct, m_incorrect", j_correct_m_incorrect_count)
print("j_incorrect, m_correct", j_incorrect_m_correct_count)
print("j_incorrect, m_incorrect", j_incorrect_m_incorrect_count)

# llama13b
# 788
# 212
# j_correct, m_correct 245
# j_correct, m_incorrect 188
# j_incorrect, m_correct 24
# j_incorrect, m_incorrect 543

'''
poly13b
665
335
j_correct, m_correct 107
j_correct, m_incorrect 173
j_incorrect, m_correct 162
j_incorrect, m_incorrect 558
'''
'''
stable13b
829
171
j_correct, m_correct 242
j_correct, m_incorrect 144
j_incorrect, m_correct 27
j_incorrect, m_incorrect 587
'''
'''
Olmo
715
285
j_correct, m_correct 265
j_correct, m_incorrect 281
j_incorrect, m_correct 4
j_incorrect, m_incorrect 450
'''