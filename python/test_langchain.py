from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import os

# Configurações
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "meus_docs"
MODEL_NAME = "llama3"

# 1. Conecta no Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# 2. Garante que a coleção existe
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE)
)

# 3. Carrega documentos
docs_path = "./docs"
documents = []
for filename in os.listdir(docs_path):
    with open(os.path.join(docs_path, filename), "r", encoding="utf-8") as f:
        documents.append(f.read())

# 4. Divide textos em chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = []
for doc in documents:
    texts.extend(splitter.split_text(doc))

# 5. Cria embeddings com Ollama
embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url="http://ollama:11434")

# 6. Salva vetores no Qdrant
vectorstore = Qdrant.from_texts(
    texts=texts,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME
)

# 7. Cria LLM e RAG
llm = Ollama(model=MODEL_NAME, base_url="http://ollama:11434")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 8. Faz pergunta
while True:
    query = input("\nPergunta: ")
    if query.lower() in ["exit", "quit"]:
        break
    resposta = qa_chain.run(query)
    print("Resposta:", resposta)