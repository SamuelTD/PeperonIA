import gradio as gr
import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 1. Configuration ---
COLLECTION_NAME = "cours_rag_collection"
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "mistral"

# --- 2. Initialize LangChain components ---
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOllama(model=LLM_MODEL)

# --- 3. Define the RAG prompt ---
template = """Tu es le créateur de pizza de génie Marco Fuso, Italien de naissance mais Français de cœur,\
et tu réponds à des questions pointues sur le monde de la pizza en te basant sur ce context :
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. Build the RAG chain ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def respond(message, history):
    """
    Handles a user message and returns updated chat history.
    """
    # Invoke the RAG chain
    answer = rag_chain.invoke(message)
    history = history or []
    history.append((message, answer))
    return history

# --- 5. Build and launch the Gradio app ---
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot RAG sur la Pizza 🍕")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Posez une question sur la pizza...", show_label=False)
    msg.submit(respond, [msg, chatbot], chatbot)

if __name__ == "__main__":
    demo.launch(share=True)  # mettez share=False en prod si besoin