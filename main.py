import re
from word_to_vec import Word2Vec_TDIDF
from k_means import ExtractiveSummary
from abstractive import BART

def clean_data(article):
   '''
   cleans data for summarization
   section: string for which dataset to use: train, validation, or test
   '''

   sentence_threshold = 5
   
   # convert decimals to a token
   article = re.sub(r'(\d)\.(\d)', r'\1[DECIMAL]\2', article)

   # split article by sentence
   article = article.split('.')

   # skip articles with lines less than threshold
   if len(article) < sentence_threshold:
      return None

   # remove parenthesis from sentences
   clean_article = []
   for line in article:

      # skip sentences less than size 2
      if len(line) <= 2:
         continue

      # remove end characters
      line = line.replace('\n', '')

      # remove parenthesis
      if '(' in line and ')' in line:
         left_index = line.index('(')
         right_index = line.index(')')
         line = line[0:left_index-1] + ' ' + line[right_index+2:]

      # remove punctuation
      line = re.sub(r'[^\w\s]', '', line)

      # remove extra spaces
      line = line.split(' ')
      line = ''.join(word + ' ' for word in line if len(word) > 0)
      line = line[:-1] + '.'

      clean_article.append(line)

   return clean_article

def generate_summaries(filename):

   # read file
   with open(filename, 'r') as f:
      doc = f.read()
   if doc is None:
      print('Enter a valid text file!')
      exit()
   
   # preprocess document
   doc = clean_data(doc)
   if doc is None:
      print('Article is not long enough!')
      exit()


   # get word2vec + tf-idf embeddings
   extractive_model = Word2Vec_TDIDF()
   extractive_model.create_embeddings(doc)

   # embeddings dictionary -> e[index] = [sentence, embedding]
   extractive_summary = ExtractiveSummary(extractive_model.embeddings)

   # join sentences for input to BART
   extractive_summary = ''.join(sent for sent in extractive_summary.summary) 

   # bart model
   abstractive_model = BART()
   abstractive_summary = abstractive_model.model(extractive_summary, max_length=300, min_length=100)[0]['summary_text']

   # remove unfinished sentences
   period = abstractive_summary.rindex('.') + 1
   abstractive_summary = abstractive_summary[:period]

   return extractive_summary, abstractive_summary

if __name__ == '__main__':
   ex, ab = generate_summaries(filename='article.txt')