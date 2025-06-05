from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import INDEX_PATH, EMBEDDING_MODEL_NAME

def get_retriever(top_k: int = 5):
    """
    저장된 벡터 인덱스를 불러와 Retriever 객체를 반환합니다.
    
    Args:
        top_k (int): 검색 시 반환할 문서 수 (기본값 5)
    
    Returns:
        retriever (BaseRetriever): LangChain용 retriever
    """
    try:
        # 1. 임베딩 모델 로드
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # 2. 저장된 FAISS 벡터 저장소 로드
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings)
        
        # 3. Retriever 생성
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

        print(f"[✓] Retriever 로드 완료. top_k={top_k}")
        return retriever

    except Exception as e:
        # 예외 처리: FAISS 로딩 또는 임베딩 모델 로딩 시 오류 발생 시
        print(f"[✗] 오류 발생: {e}")
        return None
