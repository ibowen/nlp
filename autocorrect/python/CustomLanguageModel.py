import math, collections

class CustomLanguageModel:

	def __init__(self, corpus):
		"""Initialize your data structures in the constructor."""
		# TODO your code here
		self.bigramCounts = collections.defaultdict(lambda: 0)
		self.unigramCounts = collections.defaultdict(lambda: 0)
		self.continuationProbs = collections.defaultdict(lambda: 0)
		self.followTypeCounts = collections.defaultdict(lambda: 0)
		self.total = 0
		self.train(corpus)
    
	def train(self, corpus):
		""" Takes a corpus and trains your language model. 
			Compute any counts or other corpus statistics in this function.
		"""  
		# TODO your code here
		for sentence in corpus.corpus:
			datums = sentence.data
			for i in range(len(datums)):
				self.total += 1
				token_curr = datums[i].word
				self.unigramCounts[token_curr] += 1
				if i > 0:
					token_prev = datums[i - 1].word
					token_key = token_prev + " " + token_curr
					self.bigramCounts[token_key] += 1
	
		for token in self.unigramCounts.keys():
			self.followTypeCounts[token] = self.followTypeCount(self.bigramCounts, token)
			self.continuationProbs[token] = self.continuationProb(self.bigramCounts, token)

	def score(self, sentence):
		""" Takes a list of strings as argument and returns the log-probability of the 
			sentence using your language model. Use whatever data you computed in train() here.
		"""
		# TODO your code here
		score = 0.0
		prob_KN = 0.0
		d = 0.75
		for i in range(len(sentence)):
			if i > 0:
				# count unigram and bigram count
				unigram_count = self.unigramCounts[sentence[i - 1]]
				token_key = sentence[i - 1] + " " + sentence[i]
				bigram_count = self.bigramCounts[token_key]
				# follow types ready for lambda
				followTypes = self.followTypeCounts[sentence[i - 1]]
				# continuation probability
				continuationprob = self.continuationProbs[sentence[i]]
				if unigram_count > 0:
					# count lambda
					lambda_i_1 = d / unigram_count * followTypes
					# final KN probability
					prob_KN = 1.0 * max((bigram_count - d), 0) / unigram_count + lambda_i_1 * continuationprob
				else:
					prob_KN = d * 0.1 * continuationprob
				score += math.log(prob_KN + 1.0e-15)    
		return score

	def followTypeCount(self, dict, startword):
		#count number of word types that follow startword
		type = 0
		keys = dict.keys()
		for key in keys:
			if key.startswith(startword + " "):
				type += 1
		return type

	def continuationProb(self, dict, endword):
		#continuation probability of endword as the novel continuation
		type = 0
		keys = dict.keys()
		for key in keys:
			if key.endswith(" " + endword):
				type += 1
		return 1.0 * type / len(dict)
