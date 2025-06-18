import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from utils import get_quantized_model, BASE_DATA_FOLDER

# Define a prompt formatting function.
# Here we train the model to complete the prompt with the correctness value.
JUDGEMENT_TRUE = "True"
JUDGEMENT_FALSE = "False"
def format_prompt(question, answer, correctness=""):
    judgement = ""
    if correctness == "Correct":
        judgement = JUDGEMENT_TRUE
    elif correctness == "Incorrect":
        judgement = JUDGEMENT_FALSE
    return (
        f"Is the answer given for the following question correct or incorrect?\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Judgement: {judgement}"
    )

def get_correct_incorrect_counts(correctness):
    """
    Count the number of correct and incorrect answers in the correctness column.
    """
    correct_count = 0
    incorrect_count = 0
    for c in correctness:
        if c == "Correct":
            correct_count += 1
        elif c == "Incorrect":
            incorrect_count += 1
    return correct_count, incorrect_count

def create_balanced_dataset(ft_df: pd.DataFrame, balance_amount: int):
    correctness = ft_df['correctness']
    answers = ft_df['answer']
    questions = ft_df['question']
    
    ft_df_data_balanced = pd.DataFrame(columns=['question', 'answer', 'correctness'])
    correct_count = 0
    incorrect_count = 0
    # Iterate through the ft_df_correctness dataframe and add it to ft_df_data_balanced. Make sure that the number of correct and incorrect are the same.
    for i, c in enumerate(correctness):
        if (c == "Correct" and correct_count < balance_amount) or (c == "Incorrect" and incorrect_count < balance_amount):
            ft_df_data_balanced = pd.concat([ft_df_data_balanced, pd.DataFrame([{'question': questions[i], 'answer': answers[i], 'correctness': correctness[i]}])], ignore_index=True)
            if c == "Correct":
                correct_count += 1
            elif c == "Incorrect":
                incorrect_count += 1
    print(ft_df_data_balanced.head(10))
    print(len(ft_df_data_balanced))
    # shuffle the rows in ft_df_data_balanced
    ft_df_data_balanced = ft_df_data_balanced.sample(frac=1).reset_index(drop=True)
    print(ft_df_data_balanced.head(10))
    print(ft_df_data_balanced.tail(10))
    return ft_df_data_balanced

def get_data_sets():
    """
    Retrieve the datasets for fine-tuning and testing.
    """
    ft_df_correctness = pd.read_csv(BASE_DATA_FOLDER + "fine_tune_data_correctness.csv")
    ft_test_df_correctness = pd.read_csv(BASE_DATA_FOLDER + "fine_tune_test_data_correctness.csv")

    correctness = ft_df_correctness['correctness']

    correct_count, incorrect_count = get_correct_incorrect_counts(correctness)
    
    print("Correct: ", correct_count)
    print("Incorrect: ", incorrect_count)


    return ft_df_correctness, ft_test_df_correctness

def fine_tune_jury(model_dir: str, model_name: str, balance_dataset: bool = False) -> None:
    """
    Fine-tune the jury model using the provided model name.
    """
    # Load the datasets
    fine_tune_data, fine_tune_test_data = get_data_sets()

    correct_count, incorrect_count = get_correct_incorrect_counts(fine_tune_data['correctness'])
    
    if balance_dataset:
        print("Balancing dataset...")
        balance_amount = min(correct_count, incorrect_count, 2000)
        # Create a balanced dataset
        fine_tune_data = create_balanced_dataset(fine_tune_data, balance_amount)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # 2. Load tokenizer and quantized model
    # -------------------------------
    # Load the tokenizer for model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = get_quantized_model(model_dir, device)

    correctness = fine_tune_data['correctness']
    answers = fine_tune_data['answer']
    questions = fine_tune_data['question']

    # Create prompts from the dataframe rows.
    prompts = [
        format_prompt(q, a, c)
        for q, a, c in zip(questions, answers, correctness)
    ]

    # Create a Hugging Face dataset from the prompts.
    dataset = Dataset.from_dict({"prompt": prompts})
    
    # -------------------------------
    # 3. Set up QLoRA with LoRA adapters
    # -------------------------------
    # Define LoRA configuration. Adjust target_modules based on the model's architecture.
    lora_config = LoraConfig(
        r=8,                      # Rank of the decomposition
        lora_alpha=32,            # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Example target modules; adjust if needed.
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the model with LoRA adapters.
    model = get_peft_model(model, lora_config)

    # -------------------------------
    # 4. Tokenize the dataset
    # -------------------------------
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(example):
        # Tokenize the entire prompt. Truncate if necessary.
        return tokenizer(example["prompt"], truncation=True, max_length=256, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -------------------------------
    # 5. Define training arguments with 8-bit optimizer
    # -------------------------------
    training_args = TrainingArguments(
        output_dir="qlora_"+model_name+"_finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # For an effective larger batch size if needed.
        num_train_epochs=1,
        fp16=True,                     # Mixed precision training
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        optim="paged_adamw_8bit",      # Use the 8-bit Adam optimizer via bitsandbytes
        report_to="none"               # Disable logging to third-party services (like WandB)
    )

    # -------------------------------
    # 6. Create Trainer and fine-tune
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Start training. This will fine-tune the LoRA adapters on your data.
    trainer.train()

    # Save the fine-tuned model and adapter weights.
    model.save_pretrained("qlora_"+model_name+"_finetuned")
