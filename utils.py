from collections import defaultdict
import math

import pandas as pd
import numpy as np
from tqdm import tqdm

STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 
    'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 
    'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 
    'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 
    'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd",
    "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 
    'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 
    'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 
    'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 
    'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", 
    "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 
    'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', 
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 
    "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', 
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", 
    "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'
])

class PreprocessorAndFeatureGenerator:

    def __init__(self, df):

        self.df = df
        self.df['text'] = self.df['text'].apply(lambda x: self.preprocess_lyrics(x))

        self.DOCUMENT_FREQUENCY_DICT = self.calculate_document_frequency_for_corpus(self.df)
        self.VOCABULARY = self.shortlist_vocabulary()
        
        self.df['text_filtered'] = self.df['text'].apply(
            lambda x: [char for char in x if char in self.VOCABULARY]
        )

        self.VOCABULARY_INDEX = {word: i for i, word in enumerate(self.VOCABULARY)}
        self.NUMBER_OF_DOCUMENTS = len(self.VOCABULARY)

        self.TFIDF_DICT = None
        self.TFIDF_MATRIX = None

    def generate_tfidf_dict(self, texts, mode=None):

        tfidf_dict = defaultdict(lambda: defaultdict(float))
        
        for index, text in tqdm(enumerate(texts)):

            word_count_in_lyric = len(text)
            
            for word in text:
                
                term_frequency = text.count(word) / word_count_in_lyric
                
                inverse_document_frequency = 1 + math.log(
                    (self.NUMBER_OF_DOCUMENTS + 1) / (self.DOCUMENT_FREQUENCY_DICT.get(word, 0) + 1)
                )
                
                tfidf_dict[index][word] = term_frequency * inverse_document_frequency
            
            if index not in tfidf_dict:
                tfidf_dict[index] = {}
        
        if mode == "inference":
            return tfidf_dict
        else:
            self.TFIDF_DICT = tfidf_dict
    
    def tfidf_to_matrix(self, tfidf_matrix, texts):

        matrix = []
        for doc_id in tqdm(tfidf_matrix):
            row = [0] * len(self.VOCABULARY_INDEX)
            for word, score in tfidf_matrix[doc_id].items():
                if word in self.VOCABULARY_INDEX:
                    row[self.VOCABULARY_INDEX[word]] = score
            
            matrix.append(row)
        
        return np.array(matrix)

    def shortlist_vocabulary(self):
        VOCABULARY = set()
        for word, frequency in self.DOCUMENT_FREQUENCY_DICT.items():
            if 1000 <= frequency <= 10000:
                VOCABULARY.add(word)
        return VOCABULARY

    def calculate_document_frequency_for_corpus(self, df):
        texts = df['text'].to_list()

        document_frequency_dict = dict()
        
        for text in texts:
            visited = set()
            for word in text:
                if word in visited:
                    continue
                else:
                    document_frequency_dict[word] = document_frequency_dict.get(word, 0) + 1
                    visited.add(word)
        
        return document_frequency_dict

    def preprocess_lyrics(self, text):
        text = text.lower()
        text = "".join([char for char in text if char.isalpha() or char == " "])
        text = text.split()
        text = [token for token in text if token not in STOPWORDS]
        return text
    

class MoodClassifier:
    def __init__(self, k, max_iters):
        self.CENTROIDS = None
        self.k = k
        self.max_iters = max_iters
    
    def kmeans(self, X, seed=None):
        
        # Initialize cluster centroids
        np.random.seed(seed)
        centroids = X[np.random.choice(range(len(X)), self.k, replace=False)]
        
        for _ in tqdm(range(self.max_iters)):
            
            clusters = np.argmin(np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
            
            for i in range(self.k):
                centroids[i] = X[clusters == i].mean(axis=0)
        
        self.CENTROIDS = centroids

        return clusters, centroids
    
    def classify(self, batch):

        batch = np.array(batch)
        return np.argmin(np.sqrt(((batch - self.CENTROIDS[:, np.newaxis])**2).sum(axis=2)), axis=0)
