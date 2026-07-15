import polars as pl
import pandas as pd
import pickle
import os, sys
import torch
from trait_weaver_nlp.nlp_embeddings import eval_embeddings

import numpy as np
import pickle
import zipfile
from rich import print
from rich.console import Console

console = Console()

console.print("Loading disease data", style="bold red")

disease_ot = pl.read_parquet("data/disease/disease.parquet").to_pandas()
print(disease_ot.head(1))

console.print("Removing GO terms", style="bold red")

disease_ot = disease_ot[~disease_ot['id'].str.startswith("GO")]


efo_terms = pd.DataFrame.from_dict(dict(zip(disease_ot['id'],disease_ot['name'])), orient='index', columns=['label'])
efo_terms['efo_id'] = efo_terms.index


console.print("Loading NLP models", style="bold red")

def load_model(model_name):
    with open("models/"+model_name+".pkl","rb") as input:
        return pickle.load(input)
    
with open("models/model_names.pkl","rb") as input:
    model_names = pickle.load(input)

print(model_names)


print(torch.__version__)
print(torch.cuda.get_device_properties())


console.print("Calculate embeddings", style="bold red")
embeddings_label = {}

model_name = 'SapBERT-from-PubMedBERT-fulltext'
loaded_model = load_model(model_name)

model = loaded_model['model']
tokenizer = loaded_model['tokenizer']
input_texts = efo_terms['label'].to_list()

embeddings_label[model_name] = eval_embeddings(
    model=model,
    tokenizer=tokenizer,
    input_texts=input_texts,
    pooling='cls',
    batch_size=32
)


embeddings_df = pl.DataFrame(efo_terms).with_columns(
    pl.DataFrame({"embeddings": list(embeddings_label[model_name])})
)

console.print("Dump and zip data", style="bold red")

embeddings_df.write_parquet("output/efo_nlp_embeddings.parquet", compression="zstd")

with zipfile.ZipFile("output/efo_nlp_embeddings.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write("output/efo_nlp_embeddings.parquet", arcname="efo_nlp_embeddings.parquet")

console.print("Success!", style="bold green")
