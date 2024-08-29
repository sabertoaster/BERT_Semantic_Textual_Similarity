import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from pypdf import PdfReader
import argparse

# Get the arguments from the command line
parser = argparse.ArgumentParser(description='Compute the similarity between two PDF files')
parser.add_argument('-a', '--pathA', type=str, help='Path to the PDF file A', required=True)
parser.add_argument('-b', '--pathB', type=str, help='Path to the PDF file B', required=True)
args = vars(parser.parse_args())


def get_all_text_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Load the PhoBERT-based Longformer model and tokenizer
model_name = 'bluenguyen/longformer-phobert-base-4096'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def preprocess_text_longformer(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=4096)
    return inputs


def get_embeddings_longformer(text):
    inputs = preprocess_text_longformer(text)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs[0]  # Correctly access the hidden states
        embeddings = hidden_states.mean(dim=1).numpy()
    return embeddings


def reduce_dimensions(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings


# Example large documents
doc1 = get_all_text_pdf(args['pathA'])
doc2 = get_all_text_pdf(args['pathB'])

embedding_doc1 = get_embeddings_longformer(doc1)
embedding_doc2 = get_embeddings_longformer(doc2)
similarity_score = cosine_similarity(embedding_doc1, embedding_doc2)[0][0]
print(similarity_score)
