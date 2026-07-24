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
#===========================================================================

console.print("Loading models", style = "bold yellow")
# load model 
path = os.path.join("models","Qwen3-Embedding-0.6B.pkl")

with open(path,"rb") as file_:
    qwen_model = pickle.load(file_) 


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
console.print("Qwen names", style="bold red")

qwen_embeddings_name = eval_sentence_embeddings(
    model=qwen_model, 
    input_texts=disease_ot['name'].to_list(),
    batch_size=2
)

console.print("qwen 2", style="bold red")

qwen_embeddings_rich = eval_sentence_embeddings(
    model=qwen_model, 
    input_texts=disease_ot['RichDesc'].to_list(),
    batch_size=2
)


#===========================================================================
# Attach embeddings to dataframe
#===========================================================================
disease_ot['qwen_rich'] = list(qwen_embeddings_rich)
disease_ot['qwen_name'] = list(qwen_embeddings_name)

#===========================================================================
# Dump results
#===========================================================================

console.print("Dumping", style="bold yellow")
disease_ot.to_parquet("output/nlp_embeddings_qwen.parquet", compression="zstd")

with zipfile.ZipFile("output/nlp_embeddings_qwen.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("output/nlp_embeddings_qwen.parquet", arcname="nlp_embeddings_qwen.parquet")

console.print("Success!", style="bold green")

