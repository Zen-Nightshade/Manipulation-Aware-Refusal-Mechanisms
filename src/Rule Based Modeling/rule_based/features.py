import numpy as np
import re

from empath import Empath

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

lexicon = Empath()

def extract_empath_features(text):
    """
    Extract empath features for a single text
    """
    features = lexicon.analyze(text, normalize=True)
    return np.array(list(features.values()))

def extract_empath_pair(p1_text, p2_text):
    """
    Extract empath features for both speakers
    """
    f1 = extract_empath_features(p1_text)
    f2 = extract_empath_features(p2_text)

    return np.concatenate([f1, f2])



analyzer = SentimentIntensityAnalyzer()
def extract_vader_features(text):
    """
    Extract sentiment scores using VADER
    """
    scores = analyzer.polarity_scores(text)

    return np.array([
        scores['pos'],
        scores['neu'],
        scores['neg'],
        scores['compound']
    ])

def extract_vader_pair(p1_text, p2_text):
    """
    Extract sentiment features for both speakers + interaction
    """
    s1 = extract_vader_features(p1_text)
    s2 = extract_vader_features(p2_text)

    # interaction features 
    diff = s1 - s2

    return np.concatenate([s1, s2, diff])


def count_word(text, word):
    return len(re.findall(rf"\b{word}\b", text.lower()))

def count_intensifiers(text):
    intensifiers = ["very", "really", "always", "never", "absolutely"]
    return sum(text.lower().count(w) for w in intensifiers)

def extract_interaction_features(p1_text, p2_text):

    # ---- basic counts ----
    you_p1 = count_word(p1_text, "you")
    you_p2 = count_word(p2_text, "you")

    i_p1 = count_word(p1_text, "i")
    i_p2 = count_word(p2_text, "i")

    len_p1 = max(len(p1_text.split()), 1)
    len_p2 = max(len(p2_text.split()), 1)

    you_ratio_p1 = you_p1 / len_p1
    you_ratio_p2 = you_p2 / len_p2

    i_ratio_p1 = i_p1 / len_p1
    i_ratio_p2 = i_p2 / len_p2

    dominance = you_ratio_p1 - you_ratio_p2

    pressure_p1 = you_ratio_p1 - i_ratio_p1
    pressure_p2 = you_ratio_p2 - i_ratio_p2

    intens_p1 = count_intensifiers(p1_text)
    intens_p2 = count_intensifiers(p2_text)

    len_diff = abs(len_p1 - len_p2)

    # ---- questions ----
    q_p1 = p1_text.count("?")
    q_p2 = p2_text.count("?")

    features = [
        you_ratio_p1,
        you_ratio_p2,
        i_ratio_p1,
        i_ratio_p2,
        dominance,
        q_p1,
        q_p2,
        pressure_p1,
        pressure_p2,
        intens_p1,
        intens_p2,
        len_diff
    ]

    return np.array(features)

