# this is an example file that shows how to run IBM model 1
from models.ibm1 import IBM1
from misc.support import train_model
from misc.vocabulary import Vocabulary
from misc.utils import get_run_var, format_experiments_info
import os, time
from misc.log import Log
from collections import OrderedDict


# for training
train_french_file_path = "data/training/all/hansards.36.2.f"
train_english_file_path = "data/training/all/hansards.36.2.e"

# for validation
val_french_file_path = "data/validation/dev.f"
val_english_file_path = "data/validation/dev.e"
val_align_file_path = "data/validation/dev.wa.nonullalign"

# for test
test_french_file_path = "data/testing/test.f"
test_english_file_path = "data/testing/test.e"
test_align_file_path = "data/testing/test.wa.nonullalign"

# common
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"
output_folder = "output/"
output_folder = os.path.join(output_folder, str(get_run_var(output_folder)))
load_params_from_file = "output/0/params.pkl"

# params
min_count = 5
iterations = 10
load_params = False
training_type = "var"
include_train_ll = True
include_test_aer = True
alpha = 1e-2

# create a log object
log = Log(output_folder)

# create vocabularies
vocab_french = Vocabulary(train_french_file_path, vocab_file_path=french_vocab_path, min_count=min_count)
vocab_english = Vocabulary(train_english_file_path, vocab_file_path=english_vocab_path, min_count=min_count)

# log parameters
par = OrderedDict((
    ('model', 'IBM1'), ('iterations', iterations), ('load_params', load_params), ('training_type', training_type),
    ('train_french_file_path', train_french_file_path), ('train_eng_file_path', train_english_file_path),
    ('alpha', alpha)
    ))
if load_params:
    par['params_file_path'] = load_params_from_file

log.write(format_experiments_info(par), include_timestamp=False)
log.write('french vocabulary size is: %d' % len(vocab_french))
log.write('english vocabulary size is: %d' % len(vocab_english))

model = IBM1(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english), training_type=training_type,
             alpha=alpha)
if load_params:
    model.load_parameters(load_params_from_file)


train_model(model, vocab_french=vocab_french, vocab_english=vocab_english, iterations=iterations, log=log,
            train_french_file_path=train_french_file_path, train_english_file_path=train_english_file_path,
            valid_french_file_path=val_french_file_path, valid_english_file_path=val_english_file_path, valid_alignment_file_path=val_align_file_path,
            test_french_file_path=test_french_file_path, test_english_file_path=test_french_file_path, test_alignment_file_path=test_align_file_path,
            predictions_path=output_folder, include_train_ll=include_train_ll, include_test_aer=include_test_aer)

model.save_parameters(output_folder)

