import numpy as np
from models.ibm1_b import IBM1
from misc.vocabulary import Vocabulary
from misc.utils import tokenize_corpora_to_ids
from misc.support import log_info, evaluate_model
from aer import read_naacl_alignments, AERSufficientStatistics

def save_params(model, to_file):
    log_info("Saving parameters to %s" % to_file)
    with open(to_file, "w+") as f:
        np.save(f, model.p_f_given_e)

def load_params(model, from_file):
    log_info("Loading parameters from %s" % from_file)
    params = np.load(from_file)
    model.p_f_given_e = params

# Model hyperparameters
num_iterations = 20
max_vocab_size = None
min_count = 5
small_dataset = False

# Data files.
french_file_path = "data/training/small/hansards.36.2.f" if small_dataset else "data/training/hansards.36.2.f"
french_validation_file_path = "data/validation/dev.f"
english_file_path = "data/training/small/hansards.36.2.e" if small_dataset else "data/training/hansards.36.2.e"
french_validation_file_path = "data/validation/dev.f"
english_validation_file_path = "data/validation/dev.e"
french_testing_file_path = "data/testing/test/test.f"
english_testing_file_path = "data/testing/test/test.e"
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"
validation_golden = 'data/validation/dev.wa.nonullalign'
testing_golden = 'data/testing/answers/test.wa.nonullalign'

# Load the vocabularies for English and French.
vocab_french = Vocabulary(french_file_path, vocab_file_path=french_vocab_path, min_count=min_count, \
        max_size=max_vocab_size)
vocab_english = Vocabulary(english_file_path, vocab_file_path=english_vocab_path, min_count=min_count, \
        max_size=max_vocab_size)

# Set up the model.
log_info("Setting up the model, French vocabulary size = %d, English vocabulary size = %d." % \
        (len(vocab_french), len(vocab_english)))
model = IBM1(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english))
log_info("Model has been set up.")

# Tokenize the French and English sentences.
parallel_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_file_path, english_file_path=english_file_path)
parallel_validation_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_validation_file_path, english_file_path=english_validation_file_path)
parallel_testing_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_testing_file_path, english_file_path=english_testing_file_path)

# Calculate the validation AER and log likelihood for the initial parameters.
validation_aer = evaluate_model(model, validation_golden, parallel_validation_corpus, predictions_file_path=None)
testing_aer = evaluate_model(model, testing_golden, parallel_testing_corpus, predictions_file_path="alignments/ibm1_it_%d.naacl"  % 0)
val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
log_likelihood = model.compute_log_likelihood(parallel_corpus)
log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f, testing_AER = %.4f" % \
        (0, num_iterations, log_likelihood, val_log_likelihood, validation_aer, testing_aer))

# Train the model for num_iterations EM steps.
log_info("Start training model.")
for it_num in range(1, num_iterations + 1):
    model.train(parallel_corpus)

    validation_aer = evaluate_model(model, validation_golden, parallel_validation_corpus, predictions_file_path=None)
    testing_aer = evaluate_model(model, testing_golden, parallel_testing_corpus, predictions_file_path="alignments/ibm1_it_%d.naacl"  % it_num)
    val_log_likelihood = model.compute_log_likelihood(parallel_validation_corpus)
    log_likelihood = model.compute_log_likelihood(parallel_corpus)
    log_info("Iteration %2d/%d: log_likelihood = %.4f, val_log_likelihood = %.4f, validation_AER = %.4f, testing_AER = %.4f" % \
            (it_num, num_iterations, log_likelihood, val_log_likelihood, validation_aer, testing_aer))
# save_params(model, "params/final_params_ibm1_%d_it.npy" % num_iterations)

log_info("Done training model.")
