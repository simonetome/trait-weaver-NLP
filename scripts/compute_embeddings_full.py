import polars as pl
import pandas as pd
import pickle
import os, sys
import torch
from trait_weaver_nlp.nlp_embeddings import eval_embeddings, eval_sentence_embeddings

import numpy as np
import pickle
import zipfile
from rich import print
from rich.console import Console

from sentence_transformers import SentenceTransformer

console = Console()
print(torch.__version__)
print(torch.cuda.get_device_properties())

#===========================================================================
# Load models
# SapBERT
# BioLORD 2023
#===========================================================================

console.print("Loading models", style = "bold yellow")
# load model 
path = os.path.join("models","BioLORD-2023.pkl")

with open(path,"rb") as file_:
    biolord_model = pickle.load(file_) 

# Load SapBERT
def load_model(model_name):
    with open("models/"+model_name+".pkl","rb") as input:
        return pickle.load(input)
    
sapBERT_model = load_model('SapBERT-from-PubMedBERT-fulltext')

#===========================================================================
# EFO OTAR diseases
#===========================================================================

console.print("Loading diseases", style = "bold yellow")
disease_ot = pl.read_parquet("data/disease/disease.parquet").to_pandas()
disease_ot = disease_ot[['id','name','description','exactSynonyms']]


#===========================================================================
# Helper function for rich prompt of an entity
#
#   entity<, also known as: [synonyms]>.< Definition: [description]>
#   
#   n.b.: parts in '<.>' are included only if [.] is non empty. 
#   Synonyms are restricted to first three max.
#
#===========================================================================

def get_rich_desc(row):
    # clamp to 3 max terms
    desc = row['name'] 
    try:
        if len(row['exactSynonyms']) > 0:
            desc += (', also known as: ' + (',').join([*row['exactSynonyms'][:3]]))
    except:
        pass
    try:
        if len(row['description']) > 0:    
            desc += ('. Definition: ' + row['description'])
    except:
        pass
    return desc

console.print("Genereting rich descriptions", style="bold yellow")
disease_ot['RichDesc'] = disease_ot.apply(get_rich_desc, axis=1)

print(disease_ot.head(2))


#===========================================================================
# Compute embeddings
#===========================================================================

console.print("Computing embeddings", style="bold yellow")
console.print("BioLORD 1", style="bold red")

biolord_embeddings_name = eval_sentence_embeddings(
    model=biolord_model, 
    input_texts=disease_ot['name'].to_list(),
    batch_size=32
)

console.print("BioLORD 2", style="bold red")

biolord_embeddings_rich = eval_sentence_embeddings(
    model=biolord_model, 
    input_texts=disease_ot['RichDesc'].to_list())


model = sapBERT_model['model']
tokenizer = sapBERT_model['tokenizer']

console.print("SapBERT 1", style="bold red")
sapbert_embeddings_name = eval_embeddings(
    model=model,
    tokenizer=tokenizer,
    input_texts=disease_ot['name'].to_list(),
    pooling='cls',
    batch_size=32
)

console.print("SapBERT 2", style="bold red")
sapbert_embeddings_rich = eval_embeddings(
    model=model,
    tokenizer=tokenizer,
    input_texts=disease_ot['RichDesc'].to_list(),
    pooling='cls',
    batch_size=8
)

#===========================================================================
# Attach embeddings to dataframe
#===========================================================================

disease_ot['SapBERT_rich'] = list(sapbert_embeddings_rich)
disease_ot['SapBERT_name'] = list(sapbert_embeddings_name)
disease_ot['BioLORD_rich'] = list(biolord_embeddings_rich)
disease_ot['BioLORD_name'] = list(biolord_embeddings_name)

#===========================================================================
# Dump results
#===========================================================================

console.print("Dumping", style="bold yellow")
disease_ot.to_parquet("output/nlp_embeddings.parquet", compression="zstd")

with zipfile.ZipFile("output/nlp_embeddings.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("output/nlp_embeddings.parquet", arcname="nlp_embeddings.parquet")

console.print("Success!", style="bold green")

