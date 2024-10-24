## Project Description
##### (https://github.com/sabertoaster/BERT_Semantic_Textual_Similarity/blob/master/EVN_AI_BERT_final.pdf)[Report]
This is a simple program to compute the cosine similar of the two pieces of document using:
- TextProcessing
- TextEmbedding
- Longformer - PhoBERT
- Cosine Similarity

## Installation
```bash
conda create -n ENV python=3.8
conda activate ENV
pip install -r requirements.txt
```

## Usage
```bash
conda activate ENV
python main.py --doc1 "path/to/doc1.pdf" --doc2 "path/to/doc2.pdf"
```

## What I learned from this project
- How to use TextProcessing to clean the text
- How to use TextEmbedding to convert the text to vector
- How to use Longformer - PhoBERT to embed the text
- How to use Cosine Similarity to compute the similarity of two vectors
<br>
