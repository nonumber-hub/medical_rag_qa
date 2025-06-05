from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 환경변수에 vLLM 서버 URL 설정 (OpenAI 호환 API 스타일)
os.environ["OPENAI_API_KEY"] = "EMPTY"  # 인증 불필요
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

llm = ChatOpenAI(
    model_name="llama-2",  # vLLM에서 설정한 모델명
    temperature=0.7
)

# FAISS 불러오기
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("your_index_path", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 질문 실행
result = qa.invoke({"query": "What are the symptoms of bronchitis?"})
print("[Answer]", result['result'])
