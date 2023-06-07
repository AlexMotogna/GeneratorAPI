import nltk
from nltk.corpus import wordnet as wn
from collections import defaultdict


def loadVocab():
    f = open("vocab.txt", 'r')
    words = f.read().splitlines()
    f.close()
    return words


def isWordInDataset(word, vocab):
    return word in vocab


def getNames(synsets):
    return [synset.name().split('.')[0] for synset in synsets]


def processSynset(synsets, newSynsets, vocab):
    for synset in synsets:
        word = synset.name().split('.')[0]

        if (isWordInDataset(word, vocab)):
            newSynsets.add(synset)


def processWord(word, vocab):
    wordSynset = wn.synsets(word)
    newSynsets = set()
    word_counts = defaultdict(float)

    # Synonyms
    for group in wn.synonyms(word):
        for synomyn in group:
            if (isWordInDataset(synomyn, vocab)):
                word_counts[synomyn] += 1

    synomyns = [w for w in word_counts if word_counts[w] >= 0]

    for synset in wordSynset:
        # Hyponyms
        processSynset(synset.hyponyms(), newSynsets, vocab)

        #Hypernyms
        processSynset(synset.hypernyms(), newSynsets, vocab)

        #Co-Hyponyms
        for hypernym in synset.hypernyms():
            processSynset(hypernym.hyponyms(), newSynsets, vocab)


    newWords = getNames(newSynsets)

    return synomyns + newWords


def generateNewTitles(title, vocab):
    words = title.split()
    tags = nltk.pos_tag(words)

    for word, tag in tags:
        if tag in ["NN", "NNS", "NNP", "VBZ", "VBP", "VBN", "VBG", "VBD", "VB"]:
            newWords = processWord(word, vocab)
            print(newWords)

    return


# print(generateNewTitles("dark night", loadVocab()))
