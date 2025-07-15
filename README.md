# Problem
The lack of reliability of LLM's is a significant issue holding back their utility. Models might hallucinate as a consequence of intra-model bias or random chance. This is especially costly in situations that require a high degree of accuracy or in reasoning models where one error on any reasoning step can throw off the whole chain of reasoning. Having a model self verify it's own answers can reduce hallucinations that occur by random chance but this does not account for intra-model biases introduced in the model's training or data collection. Furthermore, there should be a way to quantify the confidence of the model's answer in order to reduce the chances of an hallucination to an acceptable level. 

# Solution
This project uses 3 smaller models (called "juries") to judge the veracity of the answers given by a larger model on Triviaqa questions. It uses Conformal Prediction to quantify each jury model's certainty in it's judgement of the base model and evaluates various adjudication processes to come up with a final verdict for the answer of the base model. This project also demonstrates the benefits of uncertainty quantification in evaluating model outputs. The following diagram displays the architecture of the system.

![Alt text](plots/JuryFile.drawio.png)

Once the base model generates an answer to a given trivia question, the question/answer pair is sent to each jury to determine the veracity of the answer. Each of the jury models generates it's own judgement (True or False) and softmax logits for the tokens True/False and corresponding calibrated confidence score.

# Purpose

This project is inspired by the Panel of Juries paper by Cohere. The Panel of Juries paper uses three LLM's to determine the accuracy of the output of a base model given a reference answer. 
For example, given a question such as "What is the capital of France?", the base model might output "Paris". The original question and its output is sent to a panel of 3 LLM's, each of which 
determines if the given answer to the question is correct or incorrect.
However, when determining the final judgement of the juries, their methodology simply uses a majority vote (ie, if 2 of out 3 juries say the answer is correct, than the answer is deemed correct)
rather than exploring other adjudication processes based on the confidence of each jury. If one jury is more certain of its response than another jury, it does not make sense to consider the 
votes of both of those juries as equal. This project explores and measures various adjudication methods that take into account the confidence of each jury. 
This can be used for model evaluation, exam grading, LLM reasoning, etc.

### Other notable differences between this project and the Panel of Juries

1) Juries in this project are finetuned to output either "True" or "False". The original paper uses a few shot prompt rather than finetuning. Using a few-shot prompt collapses the output space such that the logits
   of "True" and "False" repeat themselves across each sample (eg, you might see softmax logits of [0.62, 0.38] for True and False respectively across many samples). This appears to be because
   the few shot pre-prompt dominates the question/answer pair such that variations in the question/answer pair only change the logits to one of a finite set of possibilities that are effectively
   determined by the pre-prompt. This is a problem because if we want to judge the confidence of the jury LLM's using logits, we shouldn't have the confidence be influenced by the pre-prompt. This is why the
   juries in this project are being finetuned and given a zero-shot prompt so that we can more accurately measure the confidence of each jury.
2) The juries in the panel of juries paper are given a reference answer against which the base model's answer is judged. This project does not provide a reference answer and expects the juries to use their own
   internal knowledge to judge the answer.

# Measuring Confidence

The calibrated confidence is measured using a method derived from the paper "Conformal Prediction with Large Language Models for Multi-Choice Question Answering". That paper uses conformal prediction to provide
a subset of answers from multiple choices while providing a probabilistic gaurantee that one of those given answers is correct. This project adapts that methodology to provide the highest confidence with which
the model can return one answer from a set of two (ie, "True" or "False"). This confidence is then used in some of the adjudication processes given below.

# Adjudication processes

This project considers the following adjudication processes as alternatives to majority voting:
1) Max polling (logits): Takes the answer of the jury that is most certain of it's judgement as determined by the highest logits (eg, if jury A says "True" with logit of 0.8 for "True" and jury B says "False" with
   logit of 0.6 for "False", then jury A takes precedence because it has a higher logit for the judgement it gave).
2) Max polling (confidence): Same as above except it uses the calibrated confidence of the juries.
3) Calibrated confidence score: Sum the calibrated confidence scores for the juries that said "True" and the juries that said "False".
4) Calibrated multiplicative score: Given the confidence of each jury, calculate the probability that the juries saying "False" are independantly wrong and the probability that the juries saying "True" are
   independently wrong. The one that is least likely to occur is considered the correct judgement.
5) Veto poll: If any of the juries say that the answer is incorrect, it is considered incorrect.

The following data for each of the adjudication processes has been created by:
1) Calibration and test data is filtered to only include data on which one of the jury models disagree's with the other two.
2) Calibration and test sets are then shuffled between each other given a random seed.
3) Results of the test data are aggregated for each seed.
4) The data on which the models were finetuned are allowed to be imbalanced.

Given seeds 0 to 9 inclusive, the results for each adjudication process is:

![Alt text](plots/jury_adjudication.png)

Calibrated confidence score, calibrated multiplicative score, max poll (confidence), and max poll (logits) all give more accurate results than majority vote. Note: Veto poll has 0 precision, recall, and F1 because for data on which there are disagreements between the jury models, the veto method always considers the answer to be false and it's about as accurate as majority vote.

# Conclusion
The results suggest that Majority Voting is a suboptimal adjudication process and that other adjudication processes may provide higher accuracy.

# Further exploration
In theory, "Max Poll (Confidence)" should give better results than "Max Poll (Logits)" because the calibrated confidence takes into account the miscalibration of the logits for each model. However, max polling on the confidence only does about as well as max polling on the logits. This needs further investigation and experimentation with different calibration methodologies (eg, calibrating on derived subsets of the data). The following graph shows significant miscalibration between the logits and the confidence

![Alt text](plots/miscalibration.png)

# Instructions

The following are the commands used to run the code on Google Colab. This includes everything from generating the data from the base model, finetuning the jury models, getting jury responses, calibrating for confidence scores, and test results.

### Importing To Colab
!git clone --branch master https://github.com/Flaagrah/JuriesDemo.git

import os

os.environ["TORCHINDUCTOR_DISABLE_CUDA_GRAPH"] = "1"

os.environ["TORCHINDUCTOR_DISABLE"] = "1"  # disables TorchInductor entirely

os.environ["TORCH_COMPILE_DISABLE"] = "1"  # disables torch.compile() backend

os.environ["HF_TOKEN"] = <Hugging Face Token>

### Importing Libraries

!pip install accelerate cohere transformers datasets bitsandbytes --quiet

!pip install --upgrade datasets huggingface_hub fsspec

### Change Working Directory

%cd JuriesDemo

All following commands assume that the root of the project is working directory.

### Generate Base Model Data

!python create_data_sets.py CohereLabs/c4ai-command-r-v01 <Hugging Face Token>

This command creates the answers to the question in the triviaqa dataset along with the correctness of the answer which is then used as a label to fine tune and test the juries.

### Fine Tune Jury Models

!python fine_tune_jury.py stabilityai/StableBeluga-13B stable13b <Hugging Face Token>

!python fine_tune_jury.py allenai/OLMo-2-1124-13B olmo13b <Hugging Face Token>

!python fine_tune_jury.py openlm-research/open_llama_13b open_llama_13b <Hugging Face Token>

These commands fine tune the jury models.

### Run Jury Models

!python run_jury_models.py qlora_stable13b_finetuned/checkpoint-313 stable13b <Hugging Face Token>

!python run_jury_models.py qlora_olmo13b_finetuned/checkpoint-313 olmo13b <Hugging Face Token>

!python run_jury_models.py qlora_open_llama_13b_finetuned/checkpoint-313 llama13b <Hugging Face Token>

These commands generate the response of the juries to the question/answer pairs from the data generated from the base model.

# Citations
### Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models
https://arxiv.org/abs/2404.18796

### Conformal Prediction with Large Language Models for Multi-Choice Question Answering
https://arxiv.org/abs/2305.18404

### Conformal Language Modelling
https://arxiv.org/abs/2306.10193

#### Some of the code is derived from
https://github.com/Varal7/conformal-language-modeling?tab=readme-ov-file

https://github.com/Varal7/clm_aux

### Generate Results

!python main.py

This script generates the data need to analyze the various adjudication processes.
