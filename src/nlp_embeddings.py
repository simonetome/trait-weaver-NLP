from typing import Dict
import torch
import numpy as np
from tqdm.notebook import tqdm
from numpy.typing import ArrayLike



def eval_embeddings(
    model: object,
    tokenizer: object,
    input_texts: ArrayLike,
    pooling: str = "cls",
    batch_size: int = 16,
) -> Dict:
    """
    For input model, tokenizer and list of texts, return the embeddings.
    ! Input must be all-valid and non empty
    
    :param model: Description
    :type model: object
    :param tokenizer: Description
    :type tokenizer: object
    :param input_texts: Description
    :type input_texts: ArrayLike
    :param pooling: Description
    :type pooling: str
    :param batch_size: Description
    :type batch_size: int
    :return: Description
    :rtype: Dict
    """

    # check if there are empty strings 
    if (np.array(input_texts) == "").any():
        raise ValueError('Empty string in the input texts')
    if (np.array(input_texts) == np.nan).any():
        raise ValueError('NaN string in the input texts')
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)
    model.eval()

    print("Processing batches...")
    print("Using CLS representation...")


    all_embs = list()
    for i in tqdm(np.arange(0, len(input_texts), batch_size)):
        
        
        batch_texts = input_texts[i : i + batch_size]

        toks = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            #add_special_tokens = True
        )
        
        toks_cuda = {k: v.to(device) for k, v in toks.items()}
        
        
        cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        
        all_embs.append(cls_rep.cpu().detach().numpy())

    all_embs = np.concatenate(all_embs, axis=0)

    return all_embs