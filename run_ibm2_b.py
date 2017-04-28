from models.ibm2_b import IBM2
from misc.vocabulary import Vocabulary
from misc.utils import tokenize_corpora_to_ids
from misc.support import log_info, evaluate_model
from aer import read_naacl_alignments, AERSufficientStatistics
import numpy as np
import argparse

def load_params(model, from_file):
    log_info("Loading parameters from %s" % from_file)
    params = np.load(from_file)
    model.p_f_given_e = params

# Parse arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--init', default="random")
args = parser.parse_args()

# Model hyperparameters
num_iterations = 20
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
validation_golden = 'data/validation/dev.wa.nonullalign'
testing_golden = 'data/testing/answers/test.wa.nonullalign'
french_testing_file_path = "data/testing/test/test.f"
english_testing_file_path = "data/testing/test/test.e"

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
parallel_testing_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_testing_file_path, english_file_path=english_testing_file_path)

# Load IBM1 parameters
if args.init == "ibm1":
    load_params(model, "params/ibm1.npy")

# Report the likelihood before training.
validation_aer = evaluate_model(model, validation_golden, parallel_validation_corpus, predictions_file_path=None)
testing_aer = evaluate_model(model, testing_golden, parallel_testing_corpus, predictions_file_path="alignments/ibm2_it_%d.naacl"  % 0)
val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
log_likelihood = model.compute_log_likelihood(parallel_corpus)
log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f, testing_AER = %.4f" % \
        (0, num_iterations, log_likelihood, val_log_likelihood, validation_aer, testing_aer))

# Train the model for num_iterations EM steps.
log_info("Start training model.")
for it_num in range(1, num_iterations + 1):
    model.train(parallel_corpus)

    # Calculate the validation AER
    validation_aer = evaluate_model(model, validation_golden, parallel_validation_corpus, predictions_file_path=None)
    testing_aer = evaluate_model(model, testing_golden, parallel_testing_corpus, predictions_file_path="alignments/ibm2_it_%d.naacl"  % it_num)
    val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
    log_likelihood = model.compute_log_likelihood(parallel_corpus)
    log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f, testing_AER = %.4f" % \
            (it_num, num_iterations, log_likelihood, val_log_likelihood, validation_aer, testing_aer))

log_info("Done training model.")
