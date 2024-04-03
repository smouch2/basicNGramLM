import math
import time
from collections import defaultdict

class NgramLanguageModel:

    def __init__(self): 
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.k = 0.01
        
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

        pass

    def predict_unigram(self, sentence):

        #remove start token
        unigram_counts = dict(self.unigram_counts)
        if '<s>' in unigram_counts :
            del unigram_counts ['<s>']
        
        log_prob = 0.0

        #sum total unique tokens for probability calculations
        numTokens = sum(unigram_counts.values())

        #store vocabSize V as variable
        vocabSize = len(unigram_counts)

        #calculate MLE = (wordCount+k)/(numTokens+k(vocabSize))
        tokens = sentence.split() + ['</s>']

        for token in tokens:
            #get word count
            wordCount = unigram_counts.get(token, 0)

            #calculate MLE with add-k smoothing
            MLE = (wordCount + self.k)/(numTokens+ (self.k * vocabSize))

            #accumulate log probability progressively through the sentence
            log_prob += math.log(MLE)

        return log_prob


    def predict_bigram(self, sentence):

        unigram_counts = dict(self.unigram_counts)
        bigram_counts = dict(self.bigram_counts)
        vocabSize = len(self.unigram_counts)

        log_prob = 0.0

        #similarly split the sentence with start and end tokens
        tokens = ['<s>'] + sentence.split() + ['</s>']
        for i in range(len(tokens)-1):

            #form bigrams in the same fashion as in the training method
            bigram = tokens[i] + '_' + tokens[i+1]

            #get word count of Wi-1 
            prevWordCount = unigram_counts.get(tokens[i], 0)
            bigramCount = bigram_counts.get(bigram, 0)

            #calculate P(Wi|Wi-1) with add-k smoothing 
            MLE = (bigramCount + self.k)/(prevWordCount+ (self.k * vocabSize))

            #accumulate log probability
            log_prob += math.log(MLE)

        return log_prob
    


    def test_perplexity(self, test_file, ngram_size='unigram'):
       
        total_log_prob = 0.0
        tokens = []
        totalTokens = 0
        
        with open(test_file, 'r', encoding='UTF-8') as file:
            for line in file:
                #adding the ' </s>' now ensures it is counted in the totalTokens accumulation
                sentence = line.strip() + ' </s>'
                
                tokens = sentence.split()

                #accumulate the num of total tokens in test corpus
                totalTokens += len(tokens)

                #vary which predict method to call according to ngram size
                #probabilities calculated with added end token
                if ngram_size =='unigram':
                    sentence_log_prob = self.predict_unigram(line.strip())
                elif ngram_size == 'bigram':
                    sentence_log_prob = self.predict_bigram(line.strip())
                else:
                    raise ValueError('unsupported ngram size')
                
                #accumulate total log probabilities line by line
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