# chatbot.py
# Import necessary modules
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever




class HybridSearch:
    def __init__(self, documents):
        self.documents = documents

        # BM25 initialization
        tokenized_corpus = [doc.split(" ") for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Sentence transformer for embeddings
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
        self.document_embeddings = self.model.encode(documents)
        
        # FAISS initialization
        self.index = faiss.IndexFlatL2(self.document_embeddings.shape[1])
        self.index.add(np.array(self.document_embeddings).astype('float32'))

    def search(self, query, top_n=10):
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split(" "))
        top_docs_indices = np.argsort(bm25_scores)[-top_n:]
        
        # Get embeddings of top documents from BM25 search
        top_docs_embeddings = [self.document_embeddings[i] for i in top_docs_indices]
        query_embedding = self.model.encode([query])

        # FAISS search on the top documents
        sub_index = faiss.IndexFlatL2(top_docs_embeddings[0].shape[0])
        sub_index.add(np.array(top_docs_embeddings).astype('float32'))
        _, sub_dense_ranked_indices = sub_index.search(np.array(query_embedding).astype('float32'), top_n)

        # Map FAISS results back to original document indices
        final_ranked_indices = [top_docs_indices[i] for i in sub_dense_ranked_indices[0]]

        # Retrieve the actual documents
        ranked_docs = [self.documents[i] for i in final_ranked_indices]

        return ranked_docs
    


# Define a prompt template for the chatbot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the questions"),
        ("user","Question:{question}")
    ]
)

# Set up the Streamlit framework
st.title('Langchain Chatbot With LLAMA2 model')  # Set the title of the Streamlit app
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
# Initialize the Ollama model
llm=Ollama(model="llama2")

# Create a chain that combines the prompt and the Ollama model
# Invoke the chain with the input text and display the output

uploaded_files = st.file_uploader("Choose text files", type="txt", accept_multiple_files=True)

if uploaded_files:
    # Combine the contents of all uploaded files
    all_text_data = ""
    for uploaded_file in uploaded_files:
        string_data = uploaded_file.getvalue().decode("utf-8")
        all_text_data += string_data + "\n\n"  # Add some spacing between files

    # Split the combined text data
    splitted_data = all_text_data.split("\n\n")

    # Create vector store using FAISS
    vectorstore = FAISS.from_texts(
        splitted_data,
        embedding=embedding
    )
    faiss_retriever = vectorstore.as_retriever()

    bm25_retriever = BM25Retriever.from_texts(
    [
        all_text_data
    ]
    
)
    bm25_retriever.k =  2

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.4, 0.6])

    # Create the chain
    chain = (
        {"context": ensemble_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get user input for the question
    question = st.text_input("QUESTIONS")

    if question:
        result = chain.invoke(question)
        st.write(result)