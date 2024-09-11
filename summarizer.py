from nltk import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import re

testString = " Being an avid lover for randomness, Casey Reas’s talk was very interesting for me. The way he presented how chaos can be applied to visual art to bring out masterpieces answered a question that I always ask myself whenever I face this kind of art, “how was the artist able to think of such art.” What makes the art really awesome is that there is no specific correct way to create it. So, what if a bit of chaos is added to this open-source artisitc world? Indeed, the idea of randomness in art is very interesting. Although I enjoyed his beautiful art works, another thing really intrigued is the way he approaches his work. As Casey Reas demonstrated in his talk, he artfully describes the elements within his work through a series of simple commands (shown in the figure below), eloquently narrating the shapes and movements that give life to his compositions. This method of breaking down the artistic process into its fundamental building blocks not only demystifies the creation of complex artworks but also provides a unique insight into the mind of the artist. It’s a testament to the power of code as a medium for artistic expression, where algorithms become brushes, and pixels become paint on the digital canvas."

def build_similarity_matrix(sentences, stopwords=None):
    sm = np.zeros([len(sentences), len(sentences)])

    for i in range (len(sentences)):
        for j in range (len(sentences)):
            if i == j:
                continue

            sm[i][j] = sentence_similarity(sentences[i], sentences[j])

    sm = get_symmetric_matrix(sm)
    norm = np.sum(sm, axis=0)
    sm_norm = np.divide(sm, norm, where=norm != 0)
    
    return sm_norm

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return core_cosine_similarity(vector1, vector2)

def core_cosine_similarity(vector1, vector2):
    return 1 - cosine_distance(vector1, vector2)

def get_symmetric_matrix(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())

def run_page_rank(similarity_matrix):

    # constants
    damping = 0.85  # damping coefficient, usually is .85
    min_diff = 1e-5  # convergence threshold
    steps = 100  # iteration steps

    pr_vector = np.array([1] * len(similarity_matrix))

    # Iteration
    previous_pr = 0
    for epoch in range(steps):
        pr_vector = (1 - damping) + damping * np.matmul(similarity_matrix, pr_vector)
        if abs(previous_pr - sum(pr_vector)) < min_diff:
            break
        else:
            previous_pr = sum(pr_vector)

    return pr_vector

# handling white spaces
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)

def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "

def normalize_whitespace(text):
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)

def get_top_sentences(pr_vector, sentences, number = 5):

    top_sentences = []

    if pr_vector is not None:

        sorted_pr = np.argsort(pr_vector)
        sorted_pr = list(sorted_pr)
        sorted_pr.reverse()

        index = 0
        for epoch in range(number):
            sent = sentences[sorted_pr[index]]
            sent = normalize_whitespace(sent)
            top_sentences.append(sent)
            index += 1

    return top_sentences

def summarize(text):
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]
    ve = run_page_rank(build_similarity_matrix(sentences))
    top = get_top_sentences(ve, sentences, 1)
    for sen in top:
        print(sen, "\n")
