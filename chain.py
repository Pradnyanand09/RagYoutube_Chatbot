from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.chains import LLMChain
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Set the environment variable if needed
os.environ["GOOGLE_API_KEY"] = api_key

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
)

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.2)

# Function to build a RAG chain for a given transcript
def build_chain(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    documents = [Document(page_content=text) for text in texts]
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableMap({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
