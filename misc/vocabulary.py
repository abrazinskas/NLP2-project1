# -*- coding: utf-8 -*-
import itertools
import numpy as np
import glob
import nltk
import os
from utils import create_folders_if_not_exist, deal_with_accents, DEFAULT_WORD_PREPROCESSOR
from special_symbols import SPECIAL_SYMBOLS
tokenize = nltk.word_tokenize

UNK_TOKEN = "<UNK>"

# a general purpose vocabulary class
class Vocabulary:
    def __init__(self, train_data_path=None, vocab_file_path=None, max_size=None, sep=' ', min_count=5,
                 add_special_symbols=True, word_preprocessor=None):
        assert vocab_file_path is not None or train_data_path is not None

        # creating vocabulary if it does not exist
        # TODO: make it in one run: building and no reading afterwards
        if not os.path.isfile(vocab_file_path):
            vocab = construct_vocabulary(train_data_path, word_preprocessor)
            write_vocabulary(vocab, vocab_file_path)

        self.index_to_word, self.word_to_index, self.freq = read_vocabulary(vocab_file_path, max_size,
                                                                                   sep=sep, min_count=min_count,
                                                                                   add_special_symbols=add_special_symbols)
        self.special_symbols = SPECIAL_SYMBOLS

    # words: a list of words
    def add_special_words(self, words):
        n = len(self.word_to_index)
        for i in range(len(words)):
            word = words[i]
            self.word_to_index[word] = n + i
            self.index_to_word.append(word)

    def get_word(self, id):
        return self.index_to_word[id]

    # generic for words and ids
    def get_count(self, word_or_id):
        id = self.get_id(word_or_id) if isinstance(word_or_id, str) or isinstance(word_or_id, unicode) else word_or_id
        return self.freq[id]

    def get_id(self, word):
        if word in self.word_to_index:
            return self.word_to_index[word]
        else:
            return self.word_to_index[UNK_TOKEN]

    def __len__(self):
        return len(self.index_to_word)


### HELPER FUNCTIONS ###

# works both for files and folders
def construct_vocabulary(data_path, word_preprocessor):
    word_preprocessor = word_preprocessor if word_preprocessor else DEFAULT_WORD_PREPROCESSOR
    freqs = nltk.FreqDist()
    print("Creating vocabulary...")
    if os.path.isdir(data_path):
        filenames = glob.glob(data_path + "/*")
    else:
        filenames = [data_path]  # that means there is only one file
    for filename in filenames:
        with open(filename) as f:
            print("- Processing file " + filename + "...")
            for i, sentence in enumerate(f):
                tokens = tokenize(deal_with_accents(sentence.strip().decode('utf-8', 'ignore').lower()))
                tokens = [word_preprocessor(word) for word in tokens]
                # throw all that are ""
                tokens = [w for w in tokens if w!=""]
                freqs.update(tokens)
    return freqs

# writes vocabulary to a file
def write_vocabulary(vocab, output_file, sep=' ', include_special_symbols=True):
    create_folders_if_not_exist(output_file)
    with open(output_file, 'w') as f:
        if isinstance(vocab, Vocabulary):
            if include_special_symbols:
                f.write("\n".join([sep.join((str(word[0]), str(int(word[1]))))
                                   for word in zip(vocab.index_to_word, vocab.freq)]))
            else:
                f.write("\n".join([sep.join((str(word[0]), str(int(word[1]))))
                               for word in zip(vocab.index_to_word, vocab.freq) if word[0] not in SPECIAL_SYMBOLS]))
        else:
            f.write("\n".join([sep.join((word[0].encode("utf-8", 'ignore'), str(word[1]))) for word in vocab.most_common()]))
    print("Vocabulary written to " + output_file)


# min_count: drop words from vocabulary that have count less than min_count
# max_size: limit the vocabulary size (use only the top 'max_size' words in the vocabulary list)
def read_vocabulary(filename, max_size=None, min_count=5, sep=' ', add_special_symbols=True):

    # Always have an <UNK> token that maps to any word we haven't seen before.
    # It gets the frequency of all words we'd otherwise ignore due to frequency
    # of the word being too low or the max_size of the vocabulary being exceeded.
    # This frequency is updated later.
    index_to_word = [UNK_TOKEN]
    freq = [-1]

    with open(filename) as f:
        unk_count = 0
        for i, word in enumerate(f):
            splt = word.strip().split(sep)
            word = splt[0]
            count = int(splt[1])

            # dropping infrequent words
            if count >= min_count and (max_size is None or i < max_size):
                freq.append(count)
                index_to_word.append(word)
            else:
                unk_count += count

        # Update the <UNK> frequency.
        freq[0] = unk_count

    word_to_index = {}
    for i, w in enumerate(index_to_word):
        if w in word_to_index:
            print" error in %s" % w
        word_to_index[w] = i

    # appending special symbols
    if add_special_symbols:
        for ss in SPECIAL_SYMBOLS:
            if ss in word_to_index:
                continue
            word_to_index[ss] = len(word_to_index)
            index_to_word.append(ss)
            freq.append(1)  # 1 to avoid problems with sub-sampling (in division)

    return index_to_word, word_to_index, np.array(freq, dtype="float32")
