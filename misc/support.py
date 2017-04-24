from utils import tokenize_corpora_to_ids
import special_symbols
from time import gmtime, strftime, time
import re
from aer import AERSufficientStatistics, read_naacl_alignments


# a support function for training models
# TODO: make log presence requirements non-mandatory
def train_model(model, vocab_french, vocab_english, train_french_file_path, train_english_file_path,
                valid_french_file_path, valid_english_file_path, alignment_path, log, iterations=10, word_preprocessor=None):
    parallel_corpus_train = tokenize_corpora_to_ids(vocab_french, vocab_english,
                                          french_file_path=train_french_file_path,
                                          english_file_path=train_english_file_path,
                                          word_preprocessor=word_preprocessor)
    parallel_corpus_valid = tokenize_corpora_to_ids(vocab_french, vocab_english,
                                          french_file_path=valid_french_file_path,
                                          english_file_path=valid_english_file_path,
                                          word_preprocessor=word_preprocessor)
    log_likelihood = model.compute_log_likelihood(parallel_corpus_valid)
    aer = evaluate_model(model, parallel_corpus=parallel_corpus_valid, alignment_path=alignment_path)

    log.write('initial valid. log-likelihood is: %.2f' % log_likelihood)
    log.write('initial valid. AER is: %f' % aer)
    print 'starting training'
    print '----------'
    start = time()
    for it in range(1, iterations+1):
        log.write("iteration nr %d" % it)
        model.train(parallel_corpus_train)
        log_likelihood = model.compute_log_likelihood(parallel_corpus_valid)
        aer = evaluate_model(model, parallel_corpus=parallel_corpus_valid, alignment_path=alignment_path)
        log.write("valid. log-likelihood is: %.2f" % log_likelihood)
        log.write("valid. AER is: %f" % aer)
        log.write_sep()
    end = time()
    log.write("training took %f minutes " % ((end - start)/60.0))


# a support function for models evaluation
def evaluate_model(model, alignment_path, parallel_corpus):

    # 1. Read in gold alignments
    gold_sets = read_naacl_alignments(alignment_path)

    # 2. Here I have the predictions of my own algorithm
    predictions = []

    for (french_sentence, english_sentence), (s, _) in zip(parallel_corpus, gold_sets):
        alignment = model.infer_alignment(french_sentence, english_sentence)
        # print french_sentence
        # print english_sentence
        # print s
        # print alignment
        temp_pred = []
        for i, a in enumerate(alignment):
            temp_pred.append((a, i+1))
        # for e_w_indx, f_w_indx in s:
        #     temp_pred.append((alignment[f_w_indx-1], f_w_indx))
        predictions.append(set(temp_pred))
        # print temp_pred
        # print ' --------- '


    # 3. Compute AER

    # first we get an object that manages sufficient statistics
    metric = AERSufficientStatistics()
    # then we iterate over the corpus
    for gold, pred in zip(gold_sets, predictions):
        metric.update(sure=gold[0], probable=gold[1], predicted=pred)
    # AER
    return metric.aer()


# tries to match the word and return a special token
def word_to_special_token(word):
    # FLOAT
    if re.match(r'^([0-9]+\.)[0-9]+$', word):
        return special_symbols.FLOAT_TOKEN
    return word

# a function for cleaning tokens/words
def word_preprocessor(word):
    word = re.sub(r'[^\w\'\-]|[\'\-\_]{2,}', "", word)
    if len(word) == 1:
        word = re.sub(r'[^\daiu]', '', word)
    return word

def log_info(log_string):
    time_string = strftime("%H:%M:%S", gmtime())
    print("%s [INFO]: %s" % (time_string, log_string))