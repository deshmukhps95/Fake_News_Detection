import nltk
import re
import unicodedata
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from string import digits
from nltk.stem import LancasterStemmer, SnowballStemmer, PorterStemmer
from autocorrect import spell


def strip_html_tags(text: str):
    try:
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
    except Exception:
        return text
    return stripped_text


def remove_accented_chars(text: str):
    try:
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
    except Exception as e:
        return text
    return text


def remove_url(sentence: str):
    patterns = [r"http\S+", r"www\S+"]
    out = sentence
    for pattern in patterns:
        out = re.sub(pattern, "", out)
    return out


def tokenize_sentence(sentence: str):
    tokens = nltk.word_tokenize(sentence)
    return " ".join(token for token in tokens)


def strip_punctuation(sentence: str):
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    sentence = regex.sub(" ", sentence)
    return sentence


def lower_case_sentence(sentence: str):
    return sentence.lower()


def remove_numbers_from_sentence(sentence: str):
    remove_digits = str.maketrans("", "", digits)
    return sentence.translate(remove_digits)


def auto_correct_spell(sentence: str):
    sentence = sentence.split()
    return " ".join(spell(word) for word in sentence)


def simple_stemmer(text: str):
    stemmer = LancasterStemmer()
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def porter_stemming(text: str):
    stemmer = PorterStemmer()
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def snowball_stemming(text: str):
    stemmer = SnowballStemmer("english")
    text = " ".join([stemmer.stem(word.lower()) for word in text.split()])
    return text


def remove_stop_words(sentence: str):
    stop_words = set(stopwords.words("english"))
    word_tokens = sentence.split()
    filtered_sentence = " ".join(w for w in word_tokens if not w in stop_words)
    return filtered_sentence


def remove_special_characters(text: str, remove_digits=False):
    pattern = r"[^a-zA-z0-9\s]" if not remove_digits else r"[^a-zA-z\s]"
    text = re.sub(pattern, "", text)
    return text


def process(sentence):
    sentence = strip_html_tags(sentence)
    sentence = remove_accented_chars(sentence)
    sentence = remove_numbers_from_sentence(sentence)
    sentence = strip_punctuation(sentence)
    sentence = remove_url(sentence)
    sentence = tokenize_sentence(sentence)
    sentence = lower_case_sentence(sentence)
    sentence = simple_stemmer(sentence)
    return sentence
