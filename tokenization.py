import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load raw data and tokenize using AutoTokenizer
def data_loader(max_len):
    data = pd.read_csv("./agnews/train.csv").dropna()
    testdata = pd.read_csv("./agnews/test.csv").dropna()

    x_temp = data['Title'] + " " + data['Description']
    y_temp = data['Class Index'].apply(lambda x: x-1).values # Classes need to begin from 0

    # Split into training and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

    x_test = testdata['Title'] + " " + testdata['Description']
    y_test = testdata['Class Index'].apply(lambda x: x-1).values # Classes need to begin from 0

    # Import pretrained BERT model
    from tensorflow.keras.utils import to_categorical
    from transformers import AutoTokenizer

    # Tokenization of input sentences (Truncate upto max_len and add special tokens at the beginning and end of sentences)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    z_train = tokenizer(text=x_train.tolist(), add_special_tokens=True, max_length=max_len, truncation=True, padding=True, 
                    return_tensors='tf', return_token_type_ids=False, return_attention_mask=True, verbose=True)

    z_val = tokenizer(text=x_val.tolist(), add_special_tokens=True, max_length=max_len, truncation=True, padding=True, 
                    return_tensors='tf', return_token_type_ids=False, return_attention_mask=True, verbose=True)

    z_test = tokenizer(text=x_test.tolist(), add_special_tokens=True, max_length=max_len, truncation=True, padding=True,
                return_tensors='tf', return_token_type_ids=False, return_attention_mask=True, verbose=True)

    return z_train, z_val, z_test, y_train, y_val, y_test