import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text):

    if not isinstance(text, str):
        text = "" if text is None else str(text)

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