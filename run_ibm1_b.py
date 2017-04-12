from models.ibm1_b import IBM1
from misc.support import word_preprocessor
from misc.vocabulary import Vocabulary
from misc.utils import tokenize_corpora_to_ids
from misc.support import log_info

# Model hyperparameters
num_iterations = 10
max_vocab_size = 10000

# Data files.
french_file_path = "data/training/small/hansards.36.2.f"
english_file_path = "data/training/small/hansards.36.2.e"
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"

# Load the vocabularies for English and French.
vocab_french = Vocabulary(french_file_path, vocab_file_path=french_vocab_path, min_count=1,
                          word_preprocessor=word_preprocessor, max_size=max_vocab_size)
vocab_english = Vocabulary(english_file_path, vocab_file_path=english_vocab_path, min_count=1,
                           word_preprocessor=word_preprocessor, max_size=max_vocab_size)

# Set up the model.
log_info("Setting up the model, French vocabulary size = %d, English vocabulary size = %d." % \
        (len(vocab_french), len(vocab_english)))
model = IBM1(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english))
log_info("Model has been set up.")

# Tokenize the French and English sentences.
parallel_corpus = tokenize_corpora_to_ids(vocab_french, vocab_english, \
        french_file_path=french_file_path, english_file_path=english_file_path, \
        word_preprocessor=word_preprocessor)

# Report the likelihood before training.
log_likelihood = model.compute_log_likelihood(parallel_corpus)
log_info("Iteration %2d/%d: log_likelihood = %.4f" % (0, num_iterations, log_likelihood))

# Train the model for num_iterations EM steps.
log_info("Start training model.")
for it_num in range(1, num_iterations + 1):
    model.train(parallel_corpus)
    log_likelihood = model.compute_log_likelihood(parallel_corpus)
    log_info("Iteration %2d/%d: log_likelihood = %.4f" % (it_num, num_iterations, log_likelihood))

log_info("Done training model.")
