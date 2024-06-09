import re
import requests
import pymupdf  # PyMuPDF library
import random
from tqdm import tqdm
import spacy
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_sm')
min_token_length = 30

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

page_stats_list = []

def get_page_stats(text):
    page_char_count = len(text)
    page_word_count = len(text.split())
    page_sentence_count_raw = len(re.split(r'[.!?]+', text))
    page_token_count = page_char_count

    stats = {
        "page_char_count": page_char_count,
        "page_word_count": page_word_count,
        "page_sentence_count_raw": page_sentence_count_raw,
        "page_token_count": page_token_count,
        "text": text
    }

    return stats

# URL of the PDF file to download
pdf_url = "https://arxiv.org/pdf/1706.03762"

def open_and_process_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with pymupdf.open(stream=response.content, filetype="pdf") as pdf_file:
            pdf_content = ""
            page_stats_list = []

            for page in pdf_file.pages():
                text = page.get_text()
                pdf_content += text

                page_stats = get_page_stats(text)
                page_stats_list.append(page_stats)

            with open("preprocessed_pdf.txt", "w", encoding="utf-8") as file:
                file.write(pdf_content)

    else:
        print(f"Error downloading the PDF file. Status code: {response.status_code}")
    return page_stats_list

pages_and_texts = open_and_process_pdf(pdf_url)

for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

def chunk_sentences(sentences, max_tokens=384):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(nlp(sentence))
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

all_chunks = []
for item in pages_and_texts:
    sentences = item["sentences"]
    chunks = chunk_sentences(sentences)
    all_chunks.extend(chunks)

# Embeddings 
embeddings = embedding_model.encode(all_chunks)
df_embeddings = pd.DataFrame({"sentence": all_chunks, "embedding": list(embeddings)})
df_embeddings.to_csv("embeddings.csv", index=False)

# Save the chunks to a file
  # with open("chunked_text.txt", "w", encoding="utf-8") as file:
  #     for chunk in all_chunks:
  #         file.write(chunk + "\n\n")

print(df_embeddings)
