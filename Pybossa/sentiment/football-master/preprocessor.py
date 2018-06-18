#!/usr/bin/env python3

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
import string
#from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pandas as pd
#from translation import bing
#from translation.exception import TranslateError

FLAGS = re.MULTILINE | re.DOTALL
tknzrwhu = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles = False)
tknzr = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles = True)


def preProcessSerie(serie):
    listatweets = []
    for line in serie:

        line = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", line, flags=0)
        
        ht = re.compile(r'http.')
        bar = re.compile(r'//*')
        punctuation = set(string.punctuation)
        stoplist = stopwords.words('english')
        pr = ["rt", "RT", "http", "..."]

        pal = tknzrwhu.tokenize(line)

        #Abajo estaba incluido if not i.isdigit()
        pal = [i for i in pal if i not in pr
            if i not in stoplist if i not in punctuation
            if not bar.search(i) if not ht.search(i)]
        #Dentro tenia tb if d.check(str(i))

        l = ""
        for p in pal:
            l = l + " " + p

        #Descomentar si se quiere traducir
        #try:
        #    l = bing(l, dst='es')
        #except TranslateError:
        #   l = l
        listatweets.append(l)
    return pd.Series(listatweets)