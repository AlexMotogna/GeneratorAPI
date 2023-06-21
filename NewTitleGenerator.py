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

        # Holonyms
        processSynset(synset.part_holonyms(), newSynsets, vocab)

        # Meronyms
        processSynset(synset.part_meronyms(), newSynsets, vocab)

        #Co-Hyponyms
        for hypernym in synset.hypernyms():
            processSynset(hypernym.hyponyms(), newSynsets, vocab)


    newWords = getNames(newSynsets)

    return synomyns + newWords


def generateNewTitles(title, vocab, titleCount):
    words = title.split()
    tags = nltk.pos_tag(words)

    newTitleMap = []

    for word, tag in tags:
        if tag in ["NN", "NNS", "NNP", "VBZ", "VBP", "VBN", "VBG", "VBD", "VB"]:
            newWords = processWord(word, vocab)
            newTitleMap.append(newWords)
        else:
            newTitleMap.append([word])

    newTitlesList = [title]

    for i in range(titleCount - 1):
        newTitle = ""
        for words in newTitleMap:
            try:
                newTitle += words[i]
            except IndexError:
                newTitle += words[0]
            newTitle += " "
        newTitlesList.append(newTitle[:-1])

    return newTitlesList


# print(generateNewTitles("dragon in fire", loadVocab(), 6))
