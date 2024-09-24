import os
import logging
import streamlit as st
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables from Render
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PERSIST_DIR = './db/gemini/'  # Replace with your actual directory

# Integrating monitoring feature through LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Probahini Chat Bot"

# Initialize chat history
history = []

# Integrating LLM
# Initialize the Gemini Pro 1.5 model
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1,
    max_tokens=None,
    convert_system_message_to_human=True,
    timeout=None,
    max_retries=2
)

# Configure Google Generative AI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if not os.path.exists(PERSIST_DIR):
    # Data Pre-processing
    pdf_loader = DirectoryLoader("data/", glob="./*.pdf", loader_cls=PyPDFLoader)
    
    try:
        pdf_documents = pdf_loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        pdfs = splitter.split_text(pdf_context)
        vectordb = Chroma.from_texts(pdfs, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()
    except Exception as e:
        logger.error(f"Error loading and processing PDF documents: {e}")
        raise
else:
    try:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    except Exception as e:
        logger.error(f"Error loading persisted vector database: {e}")
        raise

# Initialize retriever and query chain
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

# Streamlit app code
st.title("Probahini - Your Menstruation Expert")

user_message = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_message:
        # Define the primary prompt for Probahini
        prompt = (
            "You are Probahini, a chatbot knowledgeable on menstrual health issues. "
            "Provide detailed and specific answers related to menstruation, health, and hygiene based on available documents. "
            "Your primary responses should be in English, but if a user asks in another language, reply in that language."
            "Always answer in single short detailed paragraph message."
        )
        
        query = f"{prompt}\n\nUser Question: {user_message}"
        
        # Query document-based response
        response = query_chain({"query": query})
        print(response)
        bot_response = response.get('result', None)

        # If still no valid response, provide a generic fallback message
        if not bot_response:
            bot_response = "I'm sorry, I couldn't retrieve an answer to your question."
        
        # Display the response
        st.write(bot_response)
    else:
        st.write("Please enter a question.")
