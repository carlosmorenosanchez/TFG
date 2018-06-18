#!/usr/bin/python3

from senpy.plugins import AnalysisPlugin, SentimentPlugin#, ShelfMixin
from senpy.models import Response, Entry, Sentiment, Results

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.svm import SVC


from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
import string
import os
import numpy as np
import pandas as pd
from functions import *



class FutSentPlugin(AnalysisPlugin):

    '''Plugin to detect football sentiment'''
    author = "Carlos Moreno Sanchez"
    version = '1'

    def activate(self, *args, **kwargs):


        #Definimos las variabales con los datasets de emociones y sentimiento

        df = pd.read_csv(self.datasetneutro)
        # Encode categorical variables
        df.loc[df["Info"]=="Neutro","Info"] = 1
        df.loc[df["Info"]=="NoNeutro","Info"] = 0


        # Define X and Y
        X = df['Tweet'].values.astype(str)
        y = df['Info'].values.astype(str)




        df2 = pd.read_csv(self.datasetpos)
        #df2.loc[df["Info"]=="Positivo","Info"] = 1
        #df2.loc[df["Info"]=="Negativo","Info"] = 0

        X_posneg = df2['Tweet'].values.astype(str)
        y_posneg = df2['Info'].values.astype(str)


        dfcol = pd.read_csv(self.datasetcolera)
        X_col = dfcol['Tweet'].values.astype(str)
        y_col = dfcol['Info'].values.astype(str)


        dfem = pd.read_csv(self.datasetemocion)
        X_em = dfem['Tweet'].values.astype(str)
        y_em = dfem['Info'].values.astype(str)


        dffeltris = pd.read_csv(self.datasetfeltris)
        X_feltris = dffeltris['Tweet'].values.astype(str)
        y_feltris = dffeltris['Info'].values.astype(str)


        #Creamos las pipelines y las entrenamos con los datasets


        self._pipelineNeutro = Pipeline([
                   ('features', FeatureUnion([
                                ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),

                                ('ngrams', Pipeline([
                                      ('count_vectorizer',  CountVectorizer(analyzer="word", max_df=0.5, ngram_range=[1,2])),
                                      ('tfidf_transformer', TfidfTransformer())
                                    ])),

                                ('lexical_stats', Pipeline([
                                                    ('stats', LexicalStats()),
                                                    ('vectors', DictVectorizer())
                                                ])),
                               ('lda', Pipeline([ 
                                        ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                                        ('lda',  LatentDirichletAllocation(n_topics=4, max_iter=5,
                                                               learning_method='online', 
                                                               learning_offset=50.,
                                                               random_state=0))
                                    ]))
           
                    ])),
                    
               
                        #('clf', MultinomialNB(alpha=.01))  # classifier
                    ('clf', SVC(C=10, gamma= 1, kernel='linear', probability=True))
                #('clf', AdaBoostClassifier(n_estimators=50, base_estimator=MultinomialNB(alpha=.01), learning_rate=1))
                #('modelknn', KNeighborsClassifier(n_neighbors = 13))
        ])

        self._pipelineNeutro.fit(X,y)





        self._pipelinePosNeg = Pipeline([
               ('features', FeatureUnion([
                            ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),

                            ('ngrams',  Pipeline([
                                      ('count_vectorizer',  CountVectorizer(analyzer="word", max_df=0.5, ngram_range=[1,2])),
                                      ('tfidf_transformer', TfidfTransformer())
                            ])),

                            ('lexical_stats', Pipeline([
                                                ('stats', LexicalStats()),
                                                ('vectors', DictVectorizer())
                                            ])),
                           ('lda', Pipeline([ 
                                    ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                                    ('lda',  LatentDirichletAllocation(n_topics=4, max_iter=5,
                                                           learning_method='online', 
                                                           learning_offset=50.,
                                                           random_state=0))
                                ]))
                   
                ])),
            
       
                #('clf', MultinomialNB(alpha=.01))  # classifier
            ('clf', SVC(C=10, gamma= 1, kernel='linear', probability=True))
        #('clf', AdaBoostClassifier(n_estimators=50, base_estimator=MultinomialNB(alpha=.01), learning_rate=1))
        #('modelknn', KNeighborsClassifier(n_neighbors = 13))
        ])



        self._pipelinePosNeg.fit(X_posneg,y_posneg)



        self._pipelineFelTris = Pipeline([
                ('features', FeatureUnion([
                    ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),

                   ('ngrams', Pipeline([
                                      ('count_vectorizer',  CountVectorizer(analyzer="word", max_df=0.5, ngram_range=[1,2])),
                                      ('tfidf_transformer', TfidfTransformer())
                            ])),
                   ('lexical_stats', Pipeline([
                                        ('stats', LexicalStats()),
                                        ('vectors', DictVectorizer())
                                    ])),
                   ('lda', Pipeline([ 
                            ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                            ('lda',  LatentDirichletAllocation(n_topics=4, max_iter=5,
                                                   learning_method='online', 
                                                   learning_offset=50.,
                                                   random_state=0))
                        ]))
 

              ])),
       

                #('clf', MultinomialNB(alpha=.01))  # classifier
            ('clf', SVC(C=10, gamma= 1, kernel='linear', probability=True))
        #('clf', AdaBoostClassifier(n_estimators=50, base_estimator=MultinomialNB(alpha=.01), learning_rate=1))
        #('modelknn', KNeighborsClassifier(n_neighbors = 13))
        ])
        self._pipelineFelTris.fit(X_feltris,y_feltris)

        self._pipelineEmocion = Pipeline([
                ('features', FeatureUnion([
                    ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),

                   ('ngrams', Pipeline([
                                      ('count_vectorizer',  CountVectorizer(analyzer="word", max_df=0.5, ngram_range=[1,2])),
                                      ('tfidf_transformer', TfidfTransformer())
                            ])),
                   ('lexical_stats', Pipeline([
                                        ('stats', LexicalStats()),
                                        ('vectors', DictVectorizer())
                                    ])),
                   ('lda', Pipeline([ 
                            ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                            ('lda',  LatentDirichletAllocation(n_topics=4, max_iter=5,
                                                   learning_method='online', 
                                                   learning_offset=50.,
                                                   random_state=0))
                        ]))
 

              ])),
       

                #('clf', MultinomialNB(alpha=.01))  # classifier
            ('clf', SVC(C=10, gamma= 1, kernel='linear', probability=True))
        #('clf', AdaBoostClassifier(n_estimators=50, base_estimator=MultinomialNB(alpha=.01), learning_rate=1))
        #('modelknn', KNeighborsClassifier(n_neighbors = 13))
        ])        
        self._pipelineEmocion.fit(X_em,y_em)

        self._pipelineColera = Pipeline([
                 ('features', FeatureUnion([
                    ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),

                   ('ngrams', Pipeline([
                                      ('count_vectorizer',  CountVectorizer(analyzer="word", max_df=0.5, ngram_range=[1,2])),
                                      ('tfidf_transformer', TfidfTransformer())
                            ])),
                   ('lexical_stats', Pipeline([
                                        ('stats', LexicalStats()),
                                        ('vectors', DictVectorizer())
                                    ])),
                   ('lda', Pipeline([ 
                            ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                            ('lda',  LatentDirichletAllocation(n_topics=4, max_iter=5,
                                                   learning_method='online', 
                                                   learning_offset=50.,
                                                   random_state=0))
                        ]))
 

              ])),
       

                #('clf', MultinomialNB(alpha=.01))  # classifier
            ('clf', SVC(C=10, gamma= 1, kernel='linear', probability=True))
        #('clf', AdaBoostClassifier(n_estimators=50, base_estimator=MultinomialNB(alpha=.01), learning_rate=1))
        #('modelknn', KNeighborsClassifier(n_neighbors = 13))
        ])

        self._pipelineColera.fit(X_col,y_col)



    def deactivate(self, *args, **kwargs):
        pass

    def analyse_entry(self, entry, params):

        text = entry["nif:isString"]
        
        print(text)

        #wordlist = custom_tokenizer(text)
        #print(text)

        #text = ""

        #for i in wordlist:
        #    text = text + i

        value = self._pipelineNeutro.predict([text])

        prediction = value[0]
        print("PREDICTIOn {}".format(prediction))  

        print("Valor prediction:")

        print(prediction)

        #Definimos las variabales con la salida

        es_neutro = False
        es_positivo = False
        es_negativo = False       
        es_sinemocion = False
        es_colera = False        
        es_tristeza = False
        es_felicidad = False


        #Para sentimiento


        if(int(prediction) == 1):
            print("Prediction 1")

            es_neutro = True
            es_positivo = False
            es_negativo = False

            entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}
        

        else:
            print("Prediction 0")

            es_neutro = False

            value2 = self._pipelinePosNeg.predict([text])
            prediction2 = value2[0]
            print("PREDICTION2 {}".format(prediction2))  

            print("Valor prediction:")

            print(prediction2)

            if(prediction2 == "Positivo"):
                print("Prediction 1")

                es_positivo = True
                es_negativo = False

                entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}

            else:

                es_positivo = False
                es_negativo = True

                entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}



        #Para emociones. Si no tiene emocion: sinemocion. 
        #                Si tiene emoción pasamos a ver si es cólera o no.
        #                Si era cólera nos quedamos aqui.
        #                Si no era cólera pasamos al último clasificador: felicidad o tristeza.


        value3 = self._pipelineEmocion.predict([text])
        predictionemocion = value3[0]




        #Para emociones. Si no tiene emocion: sinemocion. 

        if(predictionemocion != "Emocion"):
            print("Prediction 1")

            es_sinemocion = True
            es_colera = False        
            es_tristeza = False
            es_felicidad = False



            entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}
        
        #Si tiene emoción pasamos a ver si es cólera o no.

        else:
            print("Prediction 0")


            es_sinemocion = False

            value4 = self._pipelineColera.predict([text])
            predictioncolera = value4[0]
        
        #Si era cólera nos quedamos aqui.

            if(predictioncolera == "Colera-Asco"):
                es_colera = True        
                es_tristeza = False
                es_felicidad = False

                entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}
        
        #Si no era cólera pasamos al último clasificador: felicidad o tristeza.

            else:
                es_colera = False        

                value5 = self._pipelineFelTris.predict([text])
                predictiontris = value5[0]
                print("PREDICTION2 {}".format(predictiontris))  

                print("Valor prediction:")

                print(predictiontris)

                if(predictiontris == "Tristeza"):
                    print("Prediction 1")

                    es_tristeza = True
                    es_felicidad = False

                    entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}

                else:

                    es_tristeza = False
                    es_felicidad = True

                    entity = {'@id':'Entity0','text':text,'es_neutro':es_neutro,'es_positivo':es_positivo, 'es_negativo':es_negativo, 'es_sinemocion':es_sinemocion, 'es_colera':es_colera, 'es_tristeza':es_tristeza, 'es_felicidad':es_felicidad}





        yield entity
