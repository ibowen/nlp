import collections, math

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
        datum = sentence.data
        for i in range(len(datum)):
            token_curr = datum[i].word
            self.unigramCounts[token_curr] += 1
            if i > 0:
                token_prev = datum[i - 1].word
                token_key = token_prev + " " + token_curr
                self.bigramCounts[token_key] += 1
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in range(len(sentence)):
        if i > 0:
			unigram_count = self.unigramCounts[sentence[i - 1]]
			
			token_key = sentence[i - 1] + " " + sentence[i]
			bigram_count = self.bigramCounts[token_key]
			
			if bigram_count > 0:
				score += math.log(bigram_count)
				score -= math.log(unigram_count)
			else:
				unigram_count = self.unigramCounts[sentence[i]]
				score += math.log(unigram_count + 1)
				score -= math.log(self.total)
				score += math.log(0.4)
    return score
