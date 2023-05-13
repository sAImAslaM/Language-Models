from collections import defaultdict

import nltk
from nltk.util import ngrams


def unigram(tokens):
    # breaking down the given corpus to individual words.
    tokens = nltk.word_tokenize(corpus)
    print("Uni-gram Model Results")

    # Counting frequency of each word.
    word_freq = nltk.FreqDist(tokens)

    # Calculate the total number of words in the corpus
    total_words = len(tokens)

    # Now conditional probability of individual word.
    # word will take key and freq will take value instance of dictionary formed using FreqDist.
    for word, freq in word_freq.items():
        prob = freq / total_words
        print(f'{word}: {prob:.4f}')

    print()


"""
This code creates a bigram language model from a sample corpus by tokenizing the corpus and counting the 
occurrences of each bigram.
The bigram model is stored in a dictionary where the keys are the first words in the 
bigrams and the values are dictionaries that store the second words and their probabilities given the first words. 
The code then normalizes the counts to get probabilities and tests the bigram model by computing the probability of a 
given word2 given a word1. 
Note that this is a simple example and in practice, you may want to preprocess the corpus 
by removing stop words, performing stemming or lemmatization, and using smoothing techniques to handle unseen 
bigrams. 
"""


def bigram(corpus):
    print("Bi-gram model Results")
    # breaking down the given corpus to individual words.
    tokens = nltk.word_tokenize(corpus)
    # Initialize a dictionary to store the bi-gram model
    bigram = {}
    # Loop over the words and count the overlapping of each bigram
    for i in range(len(tokens) - 1):
        word1 = tokens[i]
        word2 = tokens[i + 1]
        if word1 in bigram:
            if word2 in bigram[word1]:
                bigram[word1][word2] += 1
            else:
                bigram[word1][word2] = 1
        else:
            bigram[word1] = {word2: 1}

    # Normalizing the probability of the words that have no probability of being together.
    for this_word in bigram:
        total_count = float(sum(bigram[this_word].values()))
        for word2 in bigram[this_word]:
            bigram[this_word][word2] /= total_count

    # Test the bigram model
    print("Test Cases")
    test1 = "I"
    test2 = "am"
    probability = bigram[test1][test2]
    print(f"The probability of '{test1}' given '{test2}' is {probability:.3f}")

    test1 = "do"
    test2 = "not"
    probability = bigram[test1][test2]
    print(f"The probability of '{test1}' given '{test2}' is {probability:.3f}")


"""This code creates a trigram language model from a sample corpus by tokenizing the corpus and counting the 
occurrences of each trigram. The trigram model is stored in a dictionary where the keys are tuples of the first two 
words in the trigrams and the values are dictionaries that store the third words and their probabilities given the 
first two words. 
The code then normalizes the counts to get probabilities and tests the trigram model by computing the probability of 
a given word3 given a word1 and word2. 
Note that this is a simple example and in practice, you may want to preprocess the corpus by removing stop words, 
performing stemming or lemmatization, and using smoothing techniques to handle unseen trigrams. """


def trigram(corpus):
    print("Tri-gram model Results")
    # breaking down the given corpus to individual words.
    tokens = nltk.word_tokenize(corpus)
    # Initialize a dictionary to store the trigram model
    trigram_model = {}
    # Loop over the words and count the overlapping of each trigram
    for i in range(len(tokens) - 2):
        word1 = tokens[i]
        word2 = tokens[i + 1]
        word3 = tokens[i + 2]
        if (word1, word2) in trigram_model:
            if word3 in trigram_model[(word1, word2)]:
                trigram_model[(word1, word2)][word3] += 1
            else:
                trigram_model[(word1, word2)][word3] = 1
        else:
            trigram_model[(word1, word2)] = {word3: 1}
    # Normalize the counts to get probabilities
    pair = (word1, word2)
    for pair in trigram_model:
        total_count = float(sum(trigram_model[(word1, word2)].values()))
        for word3 in trigram_model[(word1, word2)]:
            trigram_model[(word1, word2)][word3] /= total_count
    # Test the trigram model
    print("Test Cases")
    test1 = "I"
    test2 = "am"
    test3 = "Sam"
    probability = trigram_model[(test1, test2)][test3]
    print(f"The probability of '{test3}' given '{test1} {test2}' is {probability:.3f}")
    test1 = "I"
    test2 = "do"
    test3 = "not"
    probability = trigram_model[(test1, test2)][test3]
    print(f"The probability of '{test3}' given '{test1} {test2}' is {probability:.3f}")


def ngram():
    # Sample dataset
    dataset = ["The quick brown fox jumps over the lazy dog",
               "This is a sample sentence for the dataset",
               "The quick brown fox is not lazy"]

    # Tokenize the dataset
    tokens = []
    for sentence in dataset:
        tokens += nltk.word_tokenize(sentence)

    # Taking Input of n
    n = 1
    while n - 1 != 0:
        n = input("Enter value of n: ")

    # Create n-grams
    n_grams = ngrams(tokens, n)

    # Initialize a dictionary to store the n-gram model
    n_gram_model = defaultdict(lambda: defaultdict(int))

    # Loop over the n-grams and count their occurrences
    for n_gram in n_grams:
        prefix = n_gram[:-1]
        suffix = n_gram[-1]
        n_gram_model[prefix][suffix] += 1

    # Normalize the counts to get probabilities
    for prefix in n_gram_model:
        total_count = float(sum(n_gram_model[prefix].values()))
        for suffix in n_gram_model[prefix]:
            n_gram_model[prefix][suffix] /= total_count

    # Test the n-gram model
    prefix = ("quick", "brown")
    suffix = "fox"
    probability = n_gram_model[prefix][suffix]
    print(f"The probability of '{suffix}' given '{prefix}' is {probability:.3f}")


corpus = "<s> I am Sam </s> <s> Sam I am </s> <s> I do not like green eggs and ham </s>"

# forming Vocabulary
vocabulary = set(corpus)
print("Vocabulary: ", vocabulary)
print()
unigram(corpus)
bigram(corpus)
trigram(corpus)
ngram()
