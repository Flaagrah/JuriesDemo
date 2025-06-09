from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import pandas as pd
import torch
import transformers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import exact_match


pre_prompt = "Please answer only the question below in one or two sentences. Do not generate any subsequent questions.\n\n"

few_shot_qa = """Q: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
A: Sinclair Lewis
Q: Where in England was Dame Judi Dench born?
A: York
Q: In which decade did Billboard magazine first publish and American hit chart?
A: 30s
Q: From which country did Angola achieve independence in 1975?
A: Portugal
Q: Which city does David Soul come from?
A: Chicago
Q: Who won Super Bowl XX?
A: Chicago Bears
Q: Which was the first European country to abolish capital punishment?
A: Norway
Q: In which country did he widespread use of ISDN begin in 1988?
A: Japan
Q: What is Bruce Willis' real first name?
A: Walter
Q: Which William wrote the novel Lord Of The Flies?
A: Golding
Q: Which innovation for the car was developed by Prince Henry of Prussia in 1911?
A: Windshield wipers
Q: How is musician William Lee Conley better known?
A: Big Bill Broonzy
Q: How is Joan Molinsky better known?
A: Joan Rivers
Q: In which branch of the arts is Patricia Neary famous?
A: Ballet
Q: Which country is Europe's largest silk producer?
A: Italy
Q: The VS-300 was a type of what?
A: Helicopter
Q: At which university did Joseph Goebbels become a doctor of philosophy?
A: Heidelberg
Q: Which prince is Queen Elizabeth II's youngest son?
A: Edward
Q: When did the founder of Jehovah's Witnesses say the world would end?
A: 1914
Q: Who found the remains of the Titanic?
A: Robert Ballard
Q: Who was the only Spice Girl not to have a middle name?
A: Posh Spice
Q: What are the international registration letters of a vehicle from Algeria?
A: DZ
Q: How did Jock die in Dallas?
A: Helicopter accident
Q: What star sign is Michael Caine?
A: Pisces
Q: Who wrote the novel Evening Class?
A: Maeve Binchy
Q: Which country does the airline Air Pacific come from?
A: Fiji
Q: In which branch of the arts does Allegra Kent work?
A: Ballet
Q: Who had a 70s No 1 hit with Billy, Don't Be A Hero?
A: Bo Donaldson & The Heywoods
Q: Banting and Best pioneered the use of what?
A: Insulin
Q: Who directed the movie La Dolce Vita?
A: Federico Fellini
Q: Which country does the airline LACSA come from?
A: Costa Rica
Q: Who directed 2001: A Space Odyssey?
A: Stanley Kubrick
Q: Which is the largest of the Japanese Volcano Islands?
A: Iwo Jima
Q: Ezzard Charles was a world champion in which sport?
A: Boxing
Q: Who was the first woman to make a solo flight across the Atlantic?
A: Amelia Earhart
Q: Which port lies between Puget Sound and Lake Washington?
A: Seattle
Q: In which city were Rotary Clubs set up in 1905?
A: Chicago
Q: Who became US Vice President when Spiro Agnew resigned?
A: Gerald Ford
Q: In which decade of the 20th century was Billy Crystal born?
A: 1940s
Q: Which George invented the Kodak roll-film camera?
A: Eastman
Q: Which series had the characters Felix Unger and Oscar Madison?
A: The Odd Couple
Q: Who along with Philips developed the CD in the late 70s?
A: Sony
Q: Where is the multinational Nestle based?
A: Switzerland
Q: Do You Know Where You're Going To? was the theme from which film?
A: Mahogany
Q: 19969 was the Chinese year of which creature?
A: Rat
Q: In the 90s how many points have been awarded for finishing second in a Grand Prix?
A: 6
Q: Stapleton international airport is in which US state?
A: Colorado
Q: What was Kevin Kline's first movie?
A: Sophie's Choice
Q: Which actor had a Doberman Pinscher called Kirk?
A: William Shatner
Q: What day of the week was the Wall Street Crash?
A: Thursday
Q: The US signed a treaty with which country to allow the construction of the Panama Canal?
A: Columbia
Q: What was Prince's last No 1 of the 80s?
A: Batdance
Q: Man In The Mirror first featured on which Michel Jackson album?
A: Bad
Q: Where was the first battle with US involvement in the Korean War?
A: Suwon
Q: On which Caribbean island did Princess Diana spend he first Christmas after her divorce was announced?
A: Barbuda
Q: In which decade was Arnold Schwarzenegger born?
A: 1950s
Q: Which musical featured the song Thank Heaven for Little Girls?
A: Gigi
Q: The Queen Elizabeth liner was destroyed by fire in the 70s in which harbour?
A: Hong Kong
Q: What breed of dog did Columbo own?
A: Basset hound
Q: What was the first movie western called?
A: Kit Carson
Q: Which Oscar-winning actress was born on exactly the same day as actress Lindsay Wagner?
A: Meryl Streep
Q: Which Amendment to the Constitution brought in prohibition in 1920?
A: 18th
Q: Which oil scandal hit the US in 1924?
A: Teapot Dome Scandal
Q: Phil Collins appeared in which Spielberg film with Robin Williams?
A: Hook
Q: {}
A: """

class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, input_length=0, stop_ids=None):
        super().__init__()
        self.stop_ids = stop_ids
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> bool:
        if self.stop_ids is None:
            return False

        output = input_ids[:, self.input_length:]

        has_stop_ids = []
        for stop_id in self.stop_ids:
            has_stop_id = torch.any(output == stop_id, dim=1)
            has_stop_ids.append(has_stop_id)
        has_stop_ids = torch.stack(has_stop_ids, dim=1)

        return (has_stop_ids.any(dim=1).all())

def get_stop_word_ids(tokenizer):
    newline_token = tokenizer.encode("\n")[-1]
    # period_token = tokenizer.encode(".")[-1]
    # comma_token = tokenizer.encode(",")[-1]

    stop_word_ids = [
        newline_token
    ]
    return stop_word_ids

def get_formatted_prompt(question):
    return pre_prompt + few_shot_qa.format(question)

def get_model_output(cand_model, tokenizer, question, device):
    prompt =  get_formatted_prompt(question)
    stop_word_ids = get_stop_word_ids(tokenizer)
    inputs_before = tokenizer(prompt, return_tensors="pt")
    attention_mask = inputs_before["attention_mask"].to(device)
    input_ids = inputs_before['input_ids'].to(device)
    stopping_criteria = transformers.StoppingCriteriaList([StoppingCriteriaSub(stop_ids=stop_word_ids, input_length=input_ids.shape[1])])
    kwargs = {
        "max_new_tokens": 100,
        "return_dict_in_generate": True,
        "output_scores": True,
        "stopping_criteria": stopping_criteria,
        "num_return_sequences": 1,
    }
    # Call the model to generate the tokens of its answer
    output_ids = cand_model.generate(
        input_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    # Decode and print the generated text
    only_output_ids = output_ids[0][0][len(input_ids[0]):]
    output_text = tokenizer.decode(only_output_ids, skip_special_tokens=True)
    return output_text

def generate_model_answers(cand_model, tokenizer, data_set, file_name, device):

    # Create a pandas dataframe with the columns 'question' and 'answer'
    df = pd.DataFrame(columns=['question', 'answer'])
    #with open(file_name, "w") as f:
    # Iterate over the calibration set
    i = 0
    results = []
    for data_point in data_set:
        print(i)
        i += 1
        output_text = get_model_output(cand_model, tokenizer, data_point['question'], device)
        new_row = pd.DataFrame([{'question': data_point['question'], 'answer': output_text, 'normalized_aliases': data_point['answer']['normalized_aliases']}])
        df = pd.concat([df, new_row], ignore_index=False)

        results.append(output_text)
    df.to_csv(file_name, index=False)
    #f.close()
    return results

def create_correctness_column(filename):
    # Read fine_tune_data.csv into a pandas dataframe
    ft_df = pd.read_csv(filename, on_bad_lines='skip', engine='python')
    # Get the normalized aliases and the answer of each question
    normalized_aliases = ft_df['normalized_aliases']
    generated_answers = ft_df['answer']

    normalized_aliases_list = normalized_aliases.tolist()
    generated_answers_list = generated_answers.tolist()

    # Apply normalize_text to every answer in the generated_answers_list
    # Create a list of strings where each string is "Correct" if the normalized_answer is in the normailzed_aliases and "Incorrect" otherwise
    correctness_list = [exact_match(a, eval(normalized_aliases[i])) for i, a in enumerate(generated_answers_list)]
    print(correctness_list)
    correctness_df = pd.DataFrame({'accurate_judgement': ['Correct' if c == 1.0 else 'Incorrect' for c in correctness_list]})

    ft_df['correctness'] = correctness_df
    print(ft_df.head())
    print(ft_df.tail(10))
    # get index of '.' in filename
    ft_df.to_csv(filename[:filename.index('.')]+'_correctness.csv', index=False)
    return correctness_df

def run_base_model(model_name: str, fine_tune_data, fine_tune_test_data, calibration_data, test_data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    generate_model_answers(model, tokenizer, fine_tune_data, "fine_tune_data.csv", device)
    generate_model_answers(model, tokenizer, fine_tune_test_data, "fine_tune_test_data.csv", device)
    generate_model_answers(model, tokenizer, calibration_data, "calibration_data.csv", device)
    generate_model_answers(model, tokenizer, test_data, "test_data.csv", device)

    create_correctness_column("fine_tune_data.csv")
    create_correctness_column("fine_tune_test_data.csv")
    create_correctness_column("calibration_data.csv")
    create_correctness_column("test_data.csv")

