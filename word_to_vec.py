from gensim.models import Word2Vec
import math

class Word2Vec_TDIDF:
   def __init__(self):
      self.name = 'wtov_tfidf'
      self.num_docs = 0
      self.word_freq = {}
      self.doc_freq = {}
      self.words = []
      self.sentence_map = {}
      self.model = None
      self.tfidf_scores = {}
      self.embeddings = {}
      

   def stats(self, doc):
      '''calculates how many times a word appears in an article and how many sentences a word appears in'''

      # each sentence in document
      for sent in doc:

         temp = []
         
         # tokenize words
         sentence = sent.split(' ')

         for word in sentence:
            
            # check that word is not a space
            if len(word) > 1:
               
               # check that word is in dictionary
               if word not in self.word_freq:
                  self.word_freq[word] = 0
               
               # store word frequency
               self.word_freq[word] += 1
               temp.append(word)


         # track how many sentences a word appears in a sentence
         s = set(temp)
         for w in s:
            if w not in self.doc_freq:
               self.doc_freq[w] = 0
            self.doc_freq[w] += 1

         # used for creating word2vec model
         if len(temp) > 0:
            self.words.append(temp)
            self.sentence_map[self.num_docs] = sent

            # added a sentence so increment count
            self.num_docs += 1

      return

   
   def create_embeddings(self, doc):
      '''combine the word2vec vectors with the tf-idf score'''

      self.stats(doc)

      # create word2vec model from words
      self.model = Word2Vec(self.words, min_count=1)
      
      # calculate tfidf score for each word
      for word in self.word_freq.keys():
         idf = math.log(self.num_docs / self.doc_freq[word])
         self.tfidf_scores[word] = self.word_freq[word] * idf


      # create embeddings
      for index, sent in enumerate(self.words):
         count = 0
         total = []
         
         # for each word in a sentence
         for word in sent:

            # count number of words in sentence
            count += 1

            # get vector
            vector = self.model.wv[word]

            # get tf-idf score
            tfidf = self.tfidf_scores[word]

            # multiply and sum together
            if len(total) == 0:
               total = vector * tfidf
            else:
               total += vector * tfidf

         # get average embedding
         total = [val / count for val in total]

         self.embeddings[index] = [self.sentence_map[index], total]

      return

