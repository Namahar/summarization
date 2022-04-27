from transformers import pipeline

class BART:
   def __init__(self):
      self.name = 'bart'
      self.model = pipeline('summarization', model='model/')

      return