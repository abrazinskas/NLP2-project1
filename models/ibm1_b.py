import numpy as np

class IBM1():

    def __init__(self, french_vocab_size, english_vocab_size):
        self.expected_counts = np.zeros((french_vocab_size, english_vocab_size))

        # Initialize parameters uniformly. We have no prior knowledge. Note that these parameters
        # are not random and therefore EM will run deterministically.
        self.p_f_given_e = np.full((french_vocab_size, english_vocab_size), 1.0 / french_vocab_size)

        self.epsilon = 1e-6

    # Performs a single E and M step over the entire given dataset.
    def train(self, parallel_corpus):

        # Make sure the expected counts matrix contains zeros.
        self.expected_counts.fill(0.)

        # Perform the expectation (E) step using the current parameters.
        for (french_sentence, english_sentence) in parallel_corpus:

            # The alignment probability for IBM1 is uniform. The NULL token has already been added to the English
            # sentence in the parallel corpus.
            alignment_prob = 1. / len(english_sentence)

            for f_j in french_sentence:

                # Compute the posterior probabilities for each possible alignment for this french word.
                posterior_probs = [alignment_prob * self.p_f_given_e[f_j, e_i] for e_i in english_sentence]
                posterior_probs /= (np.sum(posterior_probs) + self.epsilon)

                for i, e_i in enumerate(english_sentence):

                    # Consider the case that f_j was generated from e_i. Add to the expected count of this event
                    # occurring, weighted by the posterior probability.
                    self.expected_counts[f_j, e_i] += posterior_probs[i]

        # Perform the maximization (M) step to update the parameters.
        self.p_f_given_e = self.expected_counts / (np.sum(self.expected_counts, axis=0, keepdims=True) + self.epsilon)

    # Computes the marginal log likelihood.
    def compute_log_likelihood(self, parallel_corpus):
        ll = 0.
        num_data_points = 0
        for (french_sentence, english_sentence) in parallel_corpus:
            num_data_points += 1

            # Alignment probability is uniform for each alignment.
            p_alignment = 1. / len(english_sentence)

            for f_j in french_sentence:
                inner_sum = 0.
                for e_i in english_sentence:
                    inner_sum += p_alignment * self.p_f_given_e[f_j, e_i]
                ll += np.log(inner_sum + 1e-10)

        return ll / num_data_points
