# this is an example file that shows how to run IBM model 1
from models.ibm1 import IBM1
from misc.support import train_model, word_preprocessor
from misc.vocabulary import Vocabulary

french_file_path = "data/training/small/hansards.36.2.f"
english_file_path = "data/training/small/hansards.36.2.e"
french_vocab_path = "data/vocabulary/french.txt"
english_vocab_path = "data/vocabulary/english.txt"

vocab_french = Vocabulary(french_file_path, vocab_file_path=french_vocab_path, min_count=1,
                          word_preprocessor=word_preprocessor)
vocab_english = Vocabulary(english_file_path, vocab_file_path=english_vocab_path, min_count=1,
                           word_preprocessor=word_preprocessor)

print 'setting up the model'
model = IBM1(french_vocab_size=len(vocab_french), english_vocab_size=len(vocab_english))
print 'done'
print '----------'

# train
train_model(model, vocab_french=vocab_french, vocab_english=vocab_english, french_file_path=french_file_path,
            english_file_path=english_file_path, word_preprocessor=word_preprocessor)