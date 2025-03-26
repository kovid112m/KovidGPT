# Standard Imports
import streamlit as st
import sounddevice as sd
import numpy as np
import os, openai, whisper, soundfile as sf
import openai
import whisper

# Langchain Imports
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Your API key for OpenAI
api_key = st.secrets["api_key"]
openai.api_key = api_key

# Load the Whisper model
stt_model = whisper.load_model("small.en")

# Set Streamlit page config
st.set_page_config(
    page_title="Hello and Welcome...",
    page_icon="🙂",
    layout="centered",
    initial_sidebar_state="auto"
)


def load_vector_store():
    path = "data/"
    # Load PDFs
    resume_loader = PyPDFLoader(os.path.join(path, "Kovid_Sharma_Resume.pdf"))
    linkedin_loader = PyPDFLoader(os.path.join(path, "Linkedin_Profile.pdf"))
    about_loader = PyPDFLoader(os.path.join(path, "About_Me.pdf"))
    resume_docs = resume_loader.load()
    linkedin_docs = linkedin_loader.load()
    about_docs = about_loader.load()

    # Split documents using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n"],
        add_start_index=True
    )
    resume_splits = splitter.split_documents(resume_docs)
    linkedin_splits = splitter.split_documents(linkedin_docs)
    about_splits = splitter.split_documents(about_docs)

    # Create embeddings and initialize vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    vector_store = InMemoryVectorStore(embedding=embeddings)
    #vector_store = Chroma(
    #    persist_directory="chroma_db_homellc2",
    #    embedding_function=embeddings
    #)

    # Add documents to the vector store (if not already indexed)
    vector_store.add_documents(resume_splits)
    vector_store.add_documents(linkedin_splits)
    vector_store.add_documents(about_splits)
    return vector_store

def initialize_chat_agent():
    # Initialize ChatOpenAI (using your chat-based model)
    my_llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key) # type: ignore

    # Define the prompt template for professional information retrieval
    prompt = PromptTemplate(
        input_variables=["query", "retrieved_data"],
        template="""You are Kovid, and you are being interviewed (Currently it is March, 2025). Answer all questions as if you are speaking personally.
                    Do not provide extra suggestions or explanations.
                    If the question requires professional details, use the provided context.
                            
                    Question: {query}
                    Context:
                    {retrieved_data}"""
    )
    # llm_chain = LLMChain(llm=my_llm, prompt=prompt)
    llm_chain = prompt | my_llm
    
    return llm_chain

# Code to record the audio
## Keeping a limit to stop recording automatically
def record_audio(duration=7, samplerate=16000):
    st.info("Recording... Speak now!")
    # Record audio as ndarray
    audio_ndarray = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    
    return audio_ndarray

def transcribe_audio(audio_array):
    st.write("Transcribing audio...")
    result = whisper.transcribe(stt_model, audio_array.flatten())
    st.write("Transcription complete!")
    return result["text"]



def chat_with_gpt(query: str):
    llm_chain = st.session_state.get("chat_agent")
    vector_store = st.session_state.get("vector_store")
    
    retriever = vector_store.as_retriever(k = 2)
    
    retrieved_docs = retriever.get_relevant_documents(query)
    retrieved_texts = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    response = llm_chain.invoke({"query": query, "retrieved_data": retrieved_texts}).content # type: ignore
    
    return response

def text_to_speech_openai(text):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice="echo",
            input=text,
            response_format="mp3"
        )
        response.write_to_file("output2.wav")
        
        data, samplerate = sf.read("output2.wav")
        sd.play(data, samplerate)
        sd.wait()                
        return "done"
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Initializing the state session with vector store and llm chain
if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
        st.write("Vector Store loaded successfully.")
    except Exception as e:
        st.error(f"Error loading vector store: {e}")

if "chat_agent" not in st.session_state:
    try:
        st.session_state.chat_agent = initialize_chat_agent()
        st.write("Chat Agent initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing chat agent: {e}")

# Setting up the Streamlit
st.title("Hello, Kovid this side :)")
st.info("Chat with me, to know more about my life")
st.info("Recording Limit: 7 seconds")

if st.button("Record & Ask"):
    audio_array = record_audio(duration=7)
    # st.audio(audio_array.flatten(), format="audio/wav", sample_rate=16000)

    # Transcribe speech to text
    text = transcribe_audio(audio_array)
    st.write("**You said:**", text)
    
    # Get AI response using the agent (agent.run is used)
    ai_response = chat_with_gpt(text) # type: ignore
    st.write("**Chatbot:**", ai_response)
    
    st.success("Hold on a second, converting to audio...")
    text_to_speech_openai(ai_response)
