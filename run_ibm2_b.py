from models.ibm2_b import IBM2
from misc.vocabulary import Vocabulary
from misc.utils import tokenize_corpora_to_ids
from misc.support import log_info
from aer import read_naacl_alignments, AERSufficientStatistics
import numpy as np
import argparse

def load_params(model, from_file):
    log_info("Loading parameters from %s" % from_file)
    params = np.load(from_file)
    model.p_f_given_e = params

def calculate_aer(predictions):
    from random import random
    # 1. Read in gold alignments
    gold_sets = read_naacl_alignments('data/validation/dev.wa.nonullalign')

    # 3. Compute AER
    metric = AERSufficientStatistics()
    for gold, pred in zip(gold_sets, predictions):
        metric.update(sure=gold[0], probable=gold[1], predicted=pred)
    return metric.aer()

# Parse arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--init', default="random")
args = parser.parse_args()

# Model hyperparameters
num_iterations = 5
max_jump = 100
max_vocab_size = None
min_count = 5
small_dataset = False

# Data files.
french_file_path = "data/training/small/hansards.36.2.f" if small_dataset else "data/training/hansards.36.2.f"
french_validation_file_path = "data/validation/dev.f"
english_file_path = "data/training/small/hansards.36.2.e" if small_dataset else "data/training/hansards.36.2.e"
english_validation_file_path = "data/validation/dev.e"
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"

# Load the vocabularies for English and French.
vocab_french = Vocabulary(french_file_path, vocab_file_path=french_vocab_path, min_count=min_count,
                          max_size=max_vocab_size)
vocab_english = Vocabulary(english_file_path, vocab_file_path=english_vocab_path, min_count=min_count,
                           max_size=max_vocab_size)

# Set up the model.
log_info("Setting up the model, French vocabulary size = %d, English vocabulary size = %d, max_jump = %d." % \
        (len(vocab_french), len(vocab_english), max_jump))
model = IBM2(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english), max_jump=max_jump, \
        init=args.init)
log_info("Model has been set up.")

# Tokenize the French and English sentences.
log_info("Loading parallel corpus from %s and %s" % (french_file_path, english_file_path))
parallel_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_file_path, english_file_path=english_file_path)
parallel_validation_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_validation_file_path, english_file_path=english_validation_file_path)

# Load IBM1 parameters
if args.init == "ibm1":
    load_params(model, "params/ibm1.npy")

# Report the likelihood before training.
predictions = []
for french_sentence, english_sentence in parallel_validation_corpus:
    alignments = model.align(french_sentence, english_sentence)

    # Remove null alignments from predictions
    filtered_alignments = [al for al in alignments if al[0] != 0]

    predictions.append(set(filtered_alignments))
aer = calculate_aer(predictions)
val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
log_likelihood = model.compute_log_likelihood(parallel_corpus)
log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f" % \
        (0, num_iterations, log_likelihood, val_log_likelihood, aer))

# Train the model for num_iterations EM steps.
log_info("Start training model.")
for it_num in range(1, num_iterations + 1):
    model.train(parallel_corpus)

    # Calculate the validation AER
    predictions = []
    for french_sentence, english_sentence in parallel_validation_corpus:
        alignments = model.align(french_sentence, english_sentence)

        # Remove null alignments from predictions
        filtered_alignments = [al for al in alignments if al[0] != 0]

        predictions.append(set(filtered_alignments))
    aer = calculate_aer(predictions)
    val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
    log_likelihood = model.compute_log_likelihood(parallel_corpus)
    log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f" % \
            (it_num, num_iterations, log_likelihood, val_log_likelihood, aer))

log_info("Done training model.")
