from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ExtractiveSummary:
   def __init__(self, embeddings):

      # kmeans parameters
      self.num_clusters = 10
      self.iterations = 10
      self.single_max_iter = 100
      self.max_sentences = 20

      self.embeddings = embeddings

      self.cosine_similarities = self.kmeans()

      self.summary = self.extractive_summary()

      return
      

   def kmeans(self):
      '''k-means algorithm for generating centroids based on article embeddings'''

      # fit model
      model = KMeans(n_clusters=self.num_clusters, 
                     init='k-means++', 
                     n_init=self.iterations, 
                     max_iter=self.single_max_iter)

      embed = [e[1] for e in self.embeddings.values()]
      embed = np.array(embed)
      model.fit(embed)

      # calculate cosine similarity
      cosine = {}
      for count, e in enumerate(embed):
         e = e.reshape(1, -1)
         sim = []

         # compare similarity between embedding and every cluster
         for cluster in model.cluster_centers_:
            cluster = cluster.reshape(1, -1)
            s = cosine_similarity(e, cluster)[0]
            sim.append(list(s))

         # take max similarity
         cosine[count] = np.max(sim)

      return cosine

   def extractive_summary(self):
      '''creates extractive summary based on cosine similarities. picks n number of sentences.'''

       # store indicies that have best cosine similarities
      indicies = []

      # convert similaries to list
      # sort similarities
      sims = sorted(self.cosine_similarities.items(), key=lambda x:x[1])

      # convert back to dict to keep sentence index mapping
      sims = dict(sims)

      # dict is ordered by values from least similiar to most similar

      # take list of keys and reverse to get sentence indicies ordered by greatest similarity to least
      keys = list(sims.keys())
      keys.reverse()

      # take best sentences
      if len(keys) > self.max_sentences:
         indicies = keys[0:self.max_sentences]

      # if sentences less than threshold use entire list
      else:
         indicies = keys.copy()

      indicies.sort()

      summary = [self.embeddings[index][0] for index in indicies]

      return summary