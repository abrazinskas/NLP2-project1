from utils import tokenize_corpora_to_ids
import special_symbols
from time import gmtime, strftime, time
import re, os
from aer import AERSufficientStatistics, read_naacl_alignments


# a support function for training models
# TODO: make the log presence requirement non-mandatory
def train_model(model, vocab_french, vocab_english, train_french_file_path, train_english_file_path,
                valid_french_file_path, valid_english_file_path, valid_alignment_file_path,
                test_french_file_path, test_english_file_path, test_alignment_file_path,
                log, iterations=10, word_preprocessor=None, predictions_path=None, include_train_ll=False,
                include_test_aer=False):
    parallel_corpus_train = tokenize_corpora_to_ids(vocab_french, vocab_english,
                                          french_file_path=train_french_file_path,
                                          english_file_path=train_english_file_path,
                                          word_preprocessor=word_preprocessor)
    parallel_corpus_valid = tokenize_corpora_to_ids(vocab_french, vocab_english,
                                          french_file_path=valid_french_file_path,
                                          english_file_path=valid_english_file_path,
                                          word_preprocessor=word_preprocessor)
    parallel_corpus_test = tokenize_corpora_to_ids(vocab_french, vocab_english,
                                          french_file_path=test_french_file_path,
                                          english_file_path=test_english_file_path,
                                          word_preprocessor=word_preprocessor)
    # valid_log_likelihood = model.compute_objective(parallel_corpus_valid)
    # valid_aer = evaluate_model(model, parallel_corpus=parallel_corpus_valid, alignment_path=valid_alignment_file_path)
    # if include_train_ll:
    #     train_obj = model.compute_objective(parallel_corpus_train)
    #     log.write('initial train. objective is: %.2f' % train_obj)
    # # log.write('initial valid. objective is: %.2f' % valid_log_likelihood)
    # log.write('initial valid. AER is: %f' % valid_aer)
    start = time()
    for it in range(1, iterations+1):
        log.write("iteration nr. %d" % it)
        model.train(parallel_corpus_train)
        # check the iteration is the last one, and if so pass a write file to collect predictions.
        valid_aer = evaluate_model(model, parallel_corpus=parallel_corpus_valid, alignment_path=valid_alignment_file_path)
        if include_train_ll:
            train_obj = model.compute_objective(parallel_corpus_train)
            log.write('train. objective is: %.2f' % train_obj)
        log.write("valid. AER is: %f" % valid_aer)
        if include_test_aer:
            test_aer = evaluate_model(model, parallel_corpus=parallel_corpus_test, alignment_path=test_alignment_file_path,
            predictions_file_path=os.path.join(predictions_path, ".".join(["test", str(it), "naacl"])))
            log.write("test AER is: %f" % test_aer)
        valid_log_likelihood = model.compute_objective(parallel_corpus_valid)
        log.write('valid. objective is: %.2f' % valid_log_likelihood)

    end = time()
    log.write("training took %f minutes " % ((end - start)/60.0))


# a support function for models evaluation, writes predictions in the naacl format if write_file_path is provided
def evaluate_model(model, alignment_path, parallel_corpus, predictions_file_path=None):

    # 1. Read in gold alignments
    gold_sets = read_naacl_alignments(alignment_path)

    # pairs are in format (e_w_indx, f_w_indx)

    # 2. Here I have the predictions of my own algorithm
    predictions = []
    sentence_number = 0
    if predictions_file_path:
        write_file = open(predictions_file_path, 'w')
    for (french_sentence, english_sentence), (s, _) in zip(parallel_corpus, gold_sets):
        sentence_number += 1
        alignment = model.infer_alignment(french_sentence, english_sentence)
        temp_pred = []
        for i, a in enumerate(alignment):
            # skip null-token alignments
            if a == 0:
                continue
            temp_pred.append((a, i+1))
            if predictions_file_path:
                write_file.write("%04d %d %d %s\n" % (sentence_number, a, i+1, "P"))
        predictions.append(set(temp_pred))

    if predictions_file_path:
        write_file.close()
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
