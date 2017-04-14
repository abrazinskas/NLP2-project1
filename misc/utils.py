# -*- coding: utf-8 -*-
import os
import errno
import unicodedata
import nltk
tokenize = nltk.word_tokenize

DEFAULT_WORD_PREPROCESSOR = lambda word: word


def create_folders_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# tokenizes corpora into clean tokens and converts them to ids based on vocabulary
# returns an array[french_sentences, english_sentences]
def tokenize_corpora_to_ids(vocab_french, vocab_english, french_file_path, english_file_path, word_preprocessor=None):
    # TODO: add null word
    word_preprocessor = word_preprocessor if word_preprocessor else DEFAULT_WORD_PREPROCESSOR
    # read french data
    collector = []
    for i, (vocab, filename) in enumerate(zip([vocab_french, vocab_english], [french_file_path, english_file_path])):
        collector.append([])
        with open(filename) as f:
            for sentence in f:
                tokens = tokenize(deal_with_accents(sentence.strip().decode('utf-8', 'ignore').lower()))
                token_ids = []
                # clean tokens and throw empty ones
                for token in tokens:
                    token = word_preprocessor(token)
                    if token != "":
                        token_ids.append(vocab.get_id(token))
                collector[i].append(token_ids)
    return collector


# removes/replaces strange symbols like é
def deal_with_accents(str):
    return unicodedata.normalize('NFD', str).encode('ascii', 'ignore')

