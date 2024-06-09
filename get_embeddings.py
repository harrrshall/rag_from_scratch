import os
import pandas as pd
min_token_length = 30

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device="cpu")

# Create a list of sentences to turn into numbers
with open("chunked_text.txt", "r", encoding="utf-8") as file:
    # read file and change sentenses to list
    sentences = file.read().splitlines()
    sentences = [sentence.strip() for sentence in sentences]
    # print(sentences[:100])
print(sentences[0])

# Sentences are encoded/embedded by calling model.encode()
embeddings = embedding_model.encode(sentences)
# Create a DataFrame to store sentences and their embeddings
df_embeddings = pd.DataFrame({"sentence": sentences, "embedding": list(embeddings)})

# Save the DataFrame to a CSV file
df_embeddings.to_csv("embeddings.csv", index=True)

# Display the DataFrame
# print(df_embeddings)
print(f"Number of sentences: {len(sentences)}")
 
