import numpy as np
from scipy.special import digamma, gammaln

class VariationalIBM1():

    def __init__(self, french_vocab_size, english_vocab_size, alpha=10e-3):
        self.expected_counts = np.zeros((french_vocab_size, english_vocab_size))

        # Initialize parameters uniformly. We have no prior knowledge. Note that these parameters
        # are not random and therefore EM will run deterministically.
        self.theta_f_given_e = np.full((french_vocab_size, english_vocab_size), 1.0 / french_vocab_size)

        self.alpha = alpha
        self.epsilon = 1e-6

    # Performs a single E and M step over the entire given dataset.
    def train(self, parallel_corpus):

        # Make sure the expected counts matrix contains zeros.
        self.expected_counts.fill(0.)

        for (french_sentence, english_sentence) in parallel_corpus:

            # The alignment probability for IBM1 is uniform. The NULL token has already been added to the English
            # sentence in the parallel corpus.
            alignment_prob = 1. / len(english_sentence)

            for f_j in french_sentence:

                # Compute the posterior probabilities for each possible alignment for this french word.
                posterior_probs = [alignment_prob * self.theta_f_given_e[f_j, e_i] for e_i in english_sentence]
                posterior_probs /= (np.sum(posterior_probs) + self.epsilon)

                for i, e_i in enumerate(english_sentence):

                    # Consider the case that f_j was generated from e_i. Add to the expected count of this event
                    # occurring, weighted by the posterior probability.
                    self.expected_counts[f_j, e_i] += posterior_probs[i]

        lambda_f_given_e = self.expected_counts + self.alpha
        self.theta_f_given_e = np.exp(digamma(lambda_f_given_e + self.epsilon) - \
                digamma(np.sum(lambda_f_given_e, axis=0, keepdims=True) + self.epsilon))

    def ELBO(self, parallel_corpus):
        ELBO = 0.
        lambda_f_given_e = self.expected_counts + self.alpha

        # Dep on parallel corpus
        for (french_sentence, english_sentence) in parallel_corpus:
            for e_i in english_sentence:
                for f_j in french_sentence:
                    ELBO += np.log(self.theta_f_given_e[f_j, e_i] + self.epsilon)

        # Indep of parallel corpus
        for e in range(self.theta_f_given_e.shape[1]):
            for f in range(self.theta_f_given_e.shape[0]):
                sum_lambda = 0.
                for f_j in french_sentence:
                    ELBO += np.log(self.theta_f_given_e[f, e] + self.epsilon) * (self.alpha - lambda_f_given_e[f, e])
                    ELBO += gammaln(lambda_f_given_e[f, e] + self.epsilon)
                    ELBO -= gammaln(self.alpha + self.epsilon)
                    sum_lambda += lambda_f_given_e[f, e]
                ELBO -= gammaln(sum_lambda + self.epsilon)
            ELBO += gammaln(self.alpha * self.theta_f_given_e.shape[0] + self.epsilon)
        return ELBO

    # Given a French and English sentence, return the Viterbi alignment, i.e. the alignment with the maximum
    # posterior probability.
    def align(self, french_sentence, english_sentence):
        alignment = np.zeros(len(french_sentence), dtype=int)
        alignment_prob = 1. / len(english_sentence)

        # Note that we can pick the best alignment individually for each French word, since the individual alignments
        # are assumed to be independent from each other in our model.
        for j, f_j in enumerate(french_sentence):
            posterior_probs = [alignment_prob * self.theta_f_given_e[f_j, e_i] for e_i in english_sentence]
            posterior_probs /= (np.sum(posterior_probs) + self.epsilon)
            alignment[j] = np.argmax(posterior_probs)
        return zip(alignment, np.arange(len(alignment)) + 1)
