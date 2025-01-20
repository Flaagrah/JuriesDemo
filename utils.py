import re
import string

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, answers):
    """True if prediction matches any answer."""
    # Remove . and \n from prediction
    prediction = re.sub(r'[.,\n]', '', prediction)
    prediction = normalize_text(prediction)
    answers = [normalize_text(a) for a in answers]
    return float(any([prediction == a for a in answers]))