# Retrieve and answer questions
from time import perf_counter as timer
from sentence_transformers import util, SentenceTransformer
import torch
import numpy as np
import pandas as pd
import textwrap

# Set OpenAI API key
from openai import OpenAI
client = OpenAI(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)


def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    return wrapped_text


# Initialize the embedding model
embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device="cpu")

# Define the query
query = "transformer model"
print(f"Query: {query}")

# Encode the query
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# Import texts and embedding DataFrame
text_chunks_and_embedding_df = pd.read_csv(
    "/home/cybernovas/Desktop/2024/RAG/embeddings.csv")

# Convert embedding column back to np.array (it got converted to string when saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding DataFrame to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

# Convert embeddings to torch tensor
embeddings = torch.tensor(np.array(
    text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32)

# Get similarity score with the dot product
start_time = timer()
dot_scores = util.dot_score(query_embedding, embeddings)[0]
end_time = timer()

print(
    f"Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

# Get the top-k results (we'll keep this to 5)
top_results_dot_product = torch.topk(dot_scores, k=5)

# Collect the top dot products and their indices
top_dot_products = []
for score, idx in zip(top_results_dot_product[0], top_results_dot_product[1]):
    top_dot_product = print_wrapped(pages_and_chunks[idx]["sentence"])
    top_dot_products.append({"text": top_dot_product, "index": idx.item()})

# Define the prompt template
prompt_template = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.

Example 1:
Query: What are the fat-soluble vitamins?
Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.

Example 2:
Query: What are the causes of type 2 diabetes?
Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.

Example 3:
Query: What is the importance of hydration for physical performance?
Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.

Now use the following context items to answer the user query:
{context}
Relevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""


def get_answer(context, query):
    # Format the prompt with the given context and query
    context_text = "\n".join([item["text"] for item in context])
    prompt_one = prompt_template.format(context=context_text, query=query)

    # Call OpenAI API to generate the answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {
            "role": "user", "content": prompt_one}],
    )

    return response.choices[0].message.content



# Example usage
context = top_dot_products
query = "what is transformer model"
answer = get_answer(context, query)
print(answer)
