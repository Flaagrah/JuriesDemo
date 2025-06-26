from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch
import pandas as pd
from jury_finetuning.run_jury import format_prompt, JUDGEMENT_TRUE, JUDGEMENT_FALSE

def judge_response(jury_model: PreTrainedModel, 
                   jury_tokenizer: PreTrainedTokenizer, 
                   question: str, 
                   answer: str, 
                   correct_token_id: int, 
                   incorrect_token_id: int) -> torch.Tensor:
    """
    Judge the response of a question using the jury model.
    :param jury_model: The jury model to use for judging.
    :param jury_tokenizer: The tokenizer for the jury model.
    :param question: The question to judge.
    :param answer: The answer to judge.
    :param correct_token_id: The token ID for the "True" judgement.
    :param incorrect_token_id: The token ID for the "False" judgement.
    """
    prompt = format_prompt(question, answer)

    # Encode the input prompt to tensors
    inputs = jury_tokenizer(prompt, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    # Generate a response
    with torch.no_grad():
        outputs = jury_model(**inputs, return_dict=True)
    logits = outputs.logits.cpu().float()
    logits = logits[:, -1, :]

    logits = logits[:, [correct_token_id, incorrect_token_id]]
    logits = torch.softmax(logits, dim=-1)
    return logits

def get_true_false_token_ids(jury_tokenizer: PreTrainedTokenizer) -> tuple[int, int]:
    """
    Get the token IDs for the true and false judgement tokens.
    """
    correct_token_id = jury_tokenizer.encode(JUDGEMENT_TRUE)[-1]
    incorrect_token_id = jury_tokenizer.encode(JUDGEMENT_FALSE)[-1]
    return correct_token_id, incorrect_token_id

def call_jury_on_single_prompt(jury_model: PreTrainedModel, 
                               jury_tokenizer: PreTrainedTokenizer, 
                               question: str, 
                               answer: str) -> list[float]:
    correct_token_id, incorrect_token_id = get_true_false_token_ids(jury_tokenizer)
    
    logits = judge_response(jury_model, jury_tokenizer, question, answer, correct_token_id, incorrect_token_id)
    logits = logits.tolist()[0]
    return logits

def call_jury(jury_model: PreTrainedModel, 
              jury_tokenizer: PreTrainedTokenizer, 
              jury_model_name: str, 
              file_name: str) -> None:
    """
    Call the jury model to judge responses from a CSV file.
    :param jury_model: The jury model to use for judging.
    :param jury_tokenizer: The tokenizer for the jury model.
    :param jury_model_name: The name of the jury model.
    :param file_name: The name of the CSV file containing questions and answers.
    """

    df_answers = pd.read_csv(file_name)
    results = []

    correct_token_id, incorrect_token_id = get_true_false_token_ids(jury_tokenizer)
    
    print("Correct token id: ", correct_token_id)
    print("Incorrect token id: ", incorrect_token_id)
    
    df = pd.DataFrame(columns=['question', 'answer', 'normalized_aliases', 'accurate_judgement', 'logits'])
    for index, row in df_answers.iterrows():
        print("Judging Response Row: ", index)
        question = row['question']
        answer = row['answer']
        normalized_aliases = row['normalized_aliases']
        accurate_judgement = row['correctness']
        logits = judge_response(jury_model, jury_tokenizer, question, answer, correct_token_id, incorrect_token_id)
        results.append(logits.tolist())
        new_row = pd.DataFrame([{'question': question, 'answer': answer, 'normalized_aliases': normalized_aliases, 'accurate_judgement': accurate_judgement, 'logits': logits.tolist()}])
        df = pd.concat([df, new_row], ignore_index=False)
    df.to_csv("jury_judgements/jury_logits_"+jury_model_name+".csv", index=False)

