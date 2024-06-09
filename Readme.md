# Rag_from_scratch
This repository contains a vanilla implementation of Retrieval-Augmented Generation (RAG) for document search and question answering. 

## Project Structure

- **chunked_text.txt**: Contains the text split into chunks for processing.
- **embeddings.csv**: Stores the embeddings generated from the text chunks.
- **preprocessed_pdf.txt**: Contains the preprocessed text extracted from the PDF.
- **get_embeddings.py**: Script to generate embeddings for the text chunks.
- **search_and_answer.py**: Script to perform search and question answering using the embeddings.
- **splitting.py**: Script to split the preprocessed text into chunks.
- **requirements.txt**: List of required packages for the project.

## Setup Instructions

To set up and run this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harrrshall/rag_from_scratch/
   cd rag_from_scratch
   
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   
3. **Run the text splitting script:**
   ```bash
   python3 splitting.py
   
4. **Generate embeddings:**
   ```bash
   python3 get_embeddings.py
   
5. **Run the search and answer script:**
   ```bash
   python3 search_and_answer.py

replace **pdf_url** in splitting.py file  line(35) to your pdf and put your openai api in search_and_answer.py file


