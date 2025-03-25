This architecture is integrated by using LangChain as the framework for constructing the chat engine in the backend,
and STREAMLIT for the frontend UI where users can interact with the chatbot.
Both these work hand in hand in python.


In this architecture:

The LLM model used is Open AI's GPT 4o.
STT Model used is OpenAI's Whisper Small.
TTS Model used is OpenAI's TTS-1 model.
An LLM prompt is defined in the code to define the behavior of the model.
In this case, the expectation from the model is to answer questions pertaining to Kovid Sharma (myself) as first person perspective.



The Retrieval Model performs RAG using **LANGCHAIN** as a framework.
The steps involved in this process is 
1. Data Ingestion using PyPDFLoaders.
2. Indexing the documents using OpenAI's "text-embedding-3-large" model.
3. Storing the indices on to ChromaDB for fast retrieval.
4. Querying on the indices to fetch relevant information.
6. Pass the user query + retrieved information to the prompt engineered LLM model
7. Return the response to user as text
8. Convert the text to speech using OpenAI's TTS-1 model and autoplay the audio on streamlit.


**DOCUMENTS ADDED : KNOWLEDGE BASE**

For simplicity, I have added my resume, linkedin profile and short description about me as pdf documents.

**DESIGN CHOICES**

**BACKEND**

LLAMA INDEX:
Llama Index streamlines the search and retrieval process through diverse set of tools that it makes available like document_loaders (PyPDFLoader).
It offers a simple framework to work with RAG.
Since this application works on documents about me, langchain is an effective framework that can quickly retrieve 
relevant document, index them and store in ChromaDB.

Additionally, it provides bunch of prompt templates to work with, that efficiently governs the LLM's behaviour allowing us to congifure the LLM to our use case.



Hence, given the ease of use and efficiency and functionality, I have chosen this framework for building the chat engine.



Vector Embeddings:

While there are plenty of embedding models available, I went ahead with OpenAI's "text-embedding-3-large".
text-embedding-3-large is ideal for Retrieval-Augmented Generation (RAG) due to its high-dimensional, semantically rich embeddings, which enhance search accuracy and retrieval relevance.
It outperforms previous models in contextual understanding and efficiency, making it well-suitable for my data.


LLM :

Among many large language models (LLMs) available, OpenAI's GPT 4o and 4 models are extremely powerful out there in the market.
While gpt-4 offers amazing features and performance, I have chosen GPT 4o as it is sufficient in our use case.

Temperature used is 0.4 - This is to help the model stay factually correct and halucanate less, but also encourages some creativity in its answres.

**FRONTEND**

**STREAMLIT** offers a seamless integration of the llama_index based chat engine to display on the frontend.
It provides methods that allows us to easily get the input from the receiver
and update the bot's chat message respectively.


This software provides options to deploy on the streamlit cloud community or run as a standalone application using the streamlit command.

Hence, given the use case of building a simple Gen AI based voice chatbot, this is an effective tool that can be used in python to work alongside langchain in the backend.
