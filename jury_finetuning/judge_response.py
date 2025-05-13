import torch
import pandas as pd
from run_jury import format_prompt, JUDGEMENT_TRUE, JUDGEMENT_FALSE

def judge_response(jury_model, jury_tokenizer, question, answer, correct_token_id, incorrect_token_id):
    prompt = format_prompt(question, answer)

    # Encode the input prompt to tensors
    inputs = jury_tokenizer(prompt, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    index = len(inputs['input_ids'][0])
    # print(inputs['input_ids'].shape)

    # Generate a response
    with torch.no_grad():
        outputs = jury_model(**inputs, return_dict=True)
    # print(outputs['logits'].shape)
    logits = outputs.logits.cpu().float()
    logits = logits[:, -1, :]

    # Convert logits to probabilities (optional)
    probabilities = torch.softmax(logits, axis=-1)
    # Predict the next token (greedy approach)
    predicted_token_id = torch.argmax(probabilities, axis=-1).numpy()[0]
    # print("prediction", predicted_token_id)
    # print decoded predicted_token_id
    # print(jury_tokenizer.decode(predicted_token_id))

    logits = logits[:, [correct_token_id, incorrect_token_id]]
    # print("Presoftmax: ", logits)
    logits = torch.softmax(logits, dim=-1)
    # print(logits)
    predicted_token_id = torch.argmax(logits, axis=-1).numpy()[0]
    # print("prediction", predicted_token_id)
    return logits

def call_jury(jury_model, jury_tokenizer, jury_model_name, file_name):

    df_answers = pd.read_csv(file_name)
    results = []

    correct_token_id = jury_tokenizer.encode(JUDGEMENT_TRUE)[-1]
    incorrect_token_id = jury_tokenizer.encode(JUDGEMENT_FALSE)[-1]
    
    print("Correct token id: ", correct_token_id)
    print("Incorrect token id: ", incorrect_token_id)
    
    df = pd.DataFrame(columns=['question', 'answer', 'normalized_aliases', 'accurate_judgement', 'logits'])
    for index, row in df_answers.iterrows():
        print(index)
        question = row['question']
        answer = row['answer']
        normalized_aliases = row['normalized_aliases']
        accurate_judgement = row['correctness']
        logits = judge_response(jury_model, jury_tokenizer, question, answer, correct_token_id, incorrect_token_id)
        results.append(logits.tolist())
        new_row = pd.DataFrame([{'question': question, 'answer': answer, 'normalized_aliases': normalized_aliases, 'accurate_judgement': accurate_judgement, 'logits': logits.tolist()}])
        df = pd.concat([df, new_row], ignore_index=False)
    df.to_csv("jury_logits_"+jury_model_name+".csv", index=False)

