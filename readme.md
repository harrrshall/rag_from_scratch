RAG basics
___________

question -> Indexing , Retrieval , Generation

RAG Advance
_________

Query Construction
Routing
Query construction
Indexing
Retrieval
Generation


________
RoadMap |
________|

Open a PDF document (you could use almost any PDF here).
Format the text of the PDF textbook ready for an embedding model (this process is known as text splitting/chunking).
Embed all of the chunks of text in the textbook and turn them into numerical representation which we can store for later.
Build a retrieval system that uses vector search to find relevant chunks of text based on a query.
Create a prompt that incorporates the retrieved pieces of text.
Generate an answer to a query based on passages from the textbook.
The above steps can broken down into two major sections:

Document preprocessing/embedding creation (steps 1-3).
Search and answer (steps 4-6).