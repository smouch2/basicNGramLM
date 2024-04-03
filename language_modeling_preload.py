import math
import time
from collections import defaultdict

class NgramLanguageModel:

    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.unigram_log_probs = {}  #dictionary init for unigram log_prob
        self.bigram_log_probs = {}   #dictionary init for bigram log_prob
        self.k = 0.01
        self.total_tokens = 0        #token var to track for normalization

    def train(self, infile='samiam.train'):        

        #take in file line by line
        with open(infile, 'r', encoding='UTF-8') as file:
                for line in file:

                    #stripping the sentance of white space and tokenizing with start and end tokens
                    sentence = line.strip()
                    tokens = ['<s>'] + line.strip().split() +['</s>']

                    #unigram counts
                    for token in tokens:
                        self.unigram_counts[token] += 1
                        
                    #bigram counts
                    for i in range(len(tokens)-1):
                        #create token_token bigrams as prescribed by instructions
                        bigram = tokens[i] + '_' + tokens[i+1]
                        self.bigram_counts[bigram] += 1

        #calculate total token count from unigram count
        self.total_tokens = sum(self.unigram_counts.values())
        vocab_size = len(self.unigram_counts)

        #calculate and fill log_prob dict for unigrams
        for unigram, count in self.unigram_counts.items():
            prob = (count + self.k) / (self.total_tokens + self.k * vocab_size)
            self.unigram_log_probs[unigram] = math.log(prob)

        #calculate and fill log_prob dict for bigrams
        for bigram, count in self.bigram_counts.items():
            first_word, _ = bigram.split('_')
            bigram_prob = (count + self.k) / (self.unigram_counts[first_word] + self.k * vocab_size)
            self.bigram_log_probs[bigram] = math.log(bigram_prob)
        pass
        

    def predict_unigram(self, sentence):
        
        log_prob = 0.0
        tokens = sentence.split() + ['</s>']
        
        #reference precalculated log_prob values and sum total log_prob
        for token in tokens:
            if token in self.unigram_log_probs:
                    log_prob += self.unigram_log_probs[token]

        return log_prob
    

    def predict_bigram(self, sentence):


        log_prob = 0.0
        tokens = ['<s>'] + sentence.split() + ['</s>']

        #reference precalculated log_prob values and sum total log_prob
        for i in range(len(tokens)-1):
            bigram = tokens[i] + '_' + tokens[i+1]
            if bigram in self.bigram_log_probs:
                log_prob += self.bigram_log_probs[bigram]

        return log_prob
    


    def test_perplexity(self, test_file, ngram_size='unigram'):
       
        total_log_prob = 0.0
        tokens = []
        totalTokens = 0
        
        with open(test_file, 'r', encoding='UTF-8') as file:
            for line in file:
                #adding the ' </s>' now ensures it is counted in the totalTokens accumulation
                sentence = line.strip() + ' </s>'
                
                #calculating probabilities for test set with included end token
                tokens = sentence.split()
                totalTokens += len(tokens)
                if ngram_size =='unigram':
                    sentence_log_prob = self.predict_unigram(line.strip())
                    
                elif ngram_size == 'bigram':
                    sentence_log_prob = self.predict_bigram(line.strip())
                else:
                    raise ValueError('unsupported ngram size')
                
                total_log_prob += sentence_log_prob

        #normalize total_log_prob
        normalized_log_prob = total_log_prob/totalTokens
        #convert normalized log space probability to perplexity
        perplexity = math.exp(-normalized_log_prob)

        return perplexity


if __name__ == '__main__':
    #measure start time
    start_time = time.time()

    ngram_lm = NgramLanguageModel()
    ngram_lm.train('samiam.train')
    print('Training perplexity, unigram:\t', ngram_lm.test_perplexity('samiam.train'))
    print('Training perplexity, bigram:\t', ngram_lm.test_perplexity('samiam.train','bigram'))
    print('Test perplexity, unigram:\t', ngram_lm.test_perplexity('samiam.test'))
    print('Test perplexity, bigram:\t', ngram_lm.test_perplexity('samiam.test','bigram'))

    #measure end time
    end_time = time.time()

    #calculate and print the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")