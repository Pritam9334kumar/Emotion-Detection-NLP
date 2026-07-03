import string
import nltk
from nltk.corpus import stopwords


try:
    stop_words = set(stopwords.words("english"))

except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = "".join(
        char
        for char in text
        if not char.isdigit()
    )

    text = "".join(
        char
        for char in text
        if char.isascii()
    )


    words = text.split()


    words = [
        word
        for word in words
        if word not in stop_words
    ]


    return " ".join(words)