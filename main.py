from fastapi import FastAPI
from rank_bm25 import BM25Okapi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

app = FastAPI()

origins = ["*"]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)

global bm25, df, stemmer, stopword

bm25 = None
df = None
stemmer = None
stopword = None

def init_bm25():
    global bm25, df, stemmer, stopword
    df = pd.read_csv('./data-mentor.csv')

    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()

    factory_stopword = StopWordRemoverFactory()
    stopword = factory_stopword.create_stop_word_remover()
    
    corpus = df['deskripsi']
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

def try_stopword(text):
    return stopword.remove(text)

def try_stemming_stopword(text):
    stemmed_text = stemmer.stem(text)
    return stopword.remove(stemmed_text)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/description")
def read_description(description: str):
    global bm25
    if bm25 is None:
        init_bm25()
        print("Initialized BM25")

    query = description
    query = try_stemming_stopword(query)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    top_k = 5
    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    top_documents = df.iloc[top_indices][['nama', 'deskripsi']]
    return top_documents.to_dict(orient='records')