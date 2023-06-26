def loadVocab():
    f = open("vocab.txt", 'r')
    words = f.read().splitlines()
    f.close()
    return words
