# -*- coding: utf-8 -*-
import os
import itertools
import errno
import unicodedata
import nltk
from special_symbols import NULL_TOKEN
tokenize = nltk.word_tokenize

DEFAULT_WORD_PREPROCESSOR = lambda word: word


def create_folders_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

# From: http://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
# Allows a generator to be looped multiple times and reset after partial iterations.
def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

# Tokenizes corpora into clean tokens and converts them to ids based on vocabulary. Returns a generate that generates
# (french, english) sentence pairs. Adds NULL tokens to the beginning of English sentences if start_null_token is
# set to true.
@multigen
def tokenize_corpora_to_ids(vocab_french, vocab_english, french_file_path, english_file_path, word_preprocessor=None, \
        start_null_token=True):
    word_preprocessor = word_preprocessor if word_preprocessor else DEFAULT_WORD_PREPROCESSOR

    with open(french_file_path) as french_file, open(english_file_path) as english_file:
        for french_sentence, english_sentence in itertools.izip(french_file, english_file):

                # Parse the French sentence
                french_tokens = tokenize(deal_with_accents(french_sentence.strip().decode('utf-8', 'ignore').lower()))
                french_token_ids = []

                # Clean tokens and throw away empty ones.
                for token in french_tokens:
                    token = word_preprocessor(token)
                    if token != "":
                        french_token_ids.append(vocab_french.get_id(token))

                # Parse the English sentence, start English sentences with the NULL token if start_null_token.
                english_tokens = tokenize(deal_with_accents(english_sentence.strip().decode('utf-8', 'ignore').lower()))
                english_token_ids = [vocab_english.get_id(NULL_TOKEN)] if start_null_token else []

                for token in english_tokens:
                    token = word_preprocessor(token)
                    if token != "":
                        english_token_ids.append(vocab_english.get_id(token))

                yield (french_token_ids, english_token_ids)

# removes/replaces strange symbols like é
def deal_with_accents(str):
    return unicodedata.normalize('NFD', str).encode('ascii', 'ignore')


def get_run_var(dir):
    if not os.path.exists(dir):
        return 0
    subdirectories = get_immediate_subdirectories(dir)
    return len(subdirectories)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def format_experiments_info(params):
    st = ""
    st += '---------------------------- \n'
    st += '---- EXPERIMENT\'S SETUP ---- \n'
    for param_name, param_value in params.iteritems():
        st += param_name + ": " + str(param_value) + '\n'
    st += '--------------------------'
    return st