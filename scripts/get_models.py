from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from typing import Dict
import pickle
import os 
from sentence_transformers import SentenceTransformer

model_names = {
    "BioClinicalBERT" : "emilyalsentzer/Bio_ClinicalBERT",
    "ClinicalBERT" : "medicalai/ClinicalBERT",
    "BioBERT" : "dmis-lab/biobert-v1.1",
    "BioClinical-ModernBERT" : "thomas-sounack/BioClinical-ModernBERT-base",
    "SapBERT-from-PubMedBERT-fulltext": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
}

sentence_models = {
    'BioLORD-2023': 'FremyCompany/BioLORD-2023',
}

def load_model(model_name: str) -> Dict[str, object]:
    try:
        if "ModernBERT" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            #model = AutoModelForMaskedLM.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

try:
    os.mkdir("models")
except:
    print("Directory models already exists")

for m in model_names: 
    
    fname = os.path.join("models",m+".pkl")
    
    if not os.path.isfile(fname):
        print("Loading: "+fname)
        model = load_model(model_names[m])
        
        with open(fname, "wb") as f:
            pickle.dump(model, f)

with open("models/model_names.pkl", "wb") as f:
           pickle.dump(model_names, f)

#=============================================
# BioLORD
#=============================================
fname = os.path.join("models","BioLORD-2023.pkl")

if not os.path.isfile(fname):
    print("Loading: "+fname)
    model = SentenceTransformer('FremyCompany/BioLORD-2023')

    with open(fname, "wb") as f:
        pickle.dump(model, f)

#===============================================
# Qwen
#===============================================
fname = os.path.join("models","Qwen3-Embedding-0.6B.pkl")

if not os.path.isfile(fname):
    print("Loading: "+fname)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    with open(fname, "wb") as f:
        pickle.dump(model, f)
