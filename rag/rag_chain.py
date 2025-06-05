from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import INDEX_PATH, EMBEDDING_MODEL_NAME, MODEL_PATHS
from transformers import pipeline

# 1. 임베딩 모델 로드
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 2. 벡터 DB 로드 (FAISS)
def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(INDEX_PATH, embeddings)

# 3. BART 모델 로드 (제너레이터로 사용)
def load_generator():
    # 파인튜닝한 BART 모델을 HuggingFace 파이프라인으로 설정
    hf_pipeline = pipeline("text2text-generation", model=MODEL_PATHS["default"])
    return HuggingFacePipeline(pipeline=hf_pipeline)

# 4. RAG 모델을 활용한 답변 생성
def generate_answer(question: str):
    try:
        # 리트리버 로드
        retriever = load_vectorstore().as_retriever(search_kwargs={"k": 5})

        # 제너레이터 모델 로드
        generator = load_generator()

        # RetrievalQA 체인 설정
        qa_chain = RetrievalQA.from_chain_type(
            llm=generator,
            retriever=retriever,
            return_source_documents=True
        )

        # 질문에 대한 답변 생성
        result = qa_chain.run(question)

        return result

    except Exception as e:
        print(f"Error generating answer: {e}")
        return None