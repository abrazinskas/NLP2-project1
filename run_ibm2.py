# this is an example file that shows how to run IBM model 1
from models.ibm2 import IBM2
from misc.support import train_model, word_to_special_token, word_preprocessor
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
align_file_path = "data/validation/dev.wa.nonullalign"

# common
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"
output_folder = "output/"
output_folder = os.path.join(output_folder, str(get_run_var(output_folder)))
load_params_from_file = "output/0/params.pkl"

# params
max_vocab_size = 50000
iterations = 10
load_params = False


# create a log object
log = Log(output_folder)

# create vocabularies
vocab_french = Vocabulary(train_french_file_path, vocab_file_path=french_vocab_path, min_count=1,
                          word_preprocessor=word_to_special_token, max_size=max_vocab_size)
vocab_english = Vocabulary(train_english_file_path, vocab_file_path=english_vocab_path, min_count=1,
                           word_preprocessor=word_to_special_token, max_size=max_vocab_size)

# log parameters
par = OrderedDict((
    ('model', 'IBM2'), ('max_vocab_size', max_vocab_size), ('iterations', iterations), ('load_params', load_params),
    ('train_french_file_path', train_french_file_path), ('train_eng_file_path', train_english_file_path)
    ))
if load_params:
    par['params_file_path'] = load_params_from_file

log.write(format_experiments_info(par))
log.write('french vocabulary size is: %d' % len(vocab_french))
log.write('english vocabulary size is: %d' % len(vocab_english))
log.write_sep()

print 'setting up the model'
model = IBM2(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english))
if load_params:
    model.load_parameters(load_params_from_file)
print 'done'
print '----------'


train_model(model, vocab_french=vocab_french, vocab_english=vocab_english, word_preprocessor=word_to_special_token,
            iterations=iterations, log=log, valid_alignment_file_path=align_file_path,
            train_french_file_path=train_french_file_path, train_english_file_path=train_english_file_path,
            valid_french_file_path=val_french_file_path, valid_english_file_path=val_english_file_path,
          )

model.save_parameters(output_folder)

