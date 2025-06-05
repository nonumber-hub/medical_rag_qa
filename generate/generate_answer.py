from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAfrom 
from langchain_community.vectorstores import FAISS
from config.settings import INDEX_PATH, EMBEDDING_MODEL_NAME
from generator_manager import GeneratorManager  # GeneratorManager 임포트

def generate_answer(question: str):
    """
    질문을 기반으로 적절한 모델을 선택하여 답변을 생성합니다.
    
    Args:
        question (str): 사용자가 입력한 질문
    
    Returns:
        result (dict): 생성된 답변과 관련 문서
    """
    # 1. 질문을 분류하여 모델 유형 결정
    model_type = classify_question(question)  # classify_question은 사용자 정의 함수
    
    # 2. 벡터스토어 로드
    vectorstore = load_vectorstore()  # load_vectorstore 함수 필요
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. 제너레이터 로드 (분류된 모델)
    generator_manager = GeneratorManager()
    generator = generator_manager.get_model(model_type)  # 모델 로드

    # 4. RetrievalQA 체인 설정
    qa_chain = RetrievalQA.from_chain_type(
        llm=generator,  # 선택된 BART 모델을 제너레이터로 사용
        retriever=retriever,
        return_source_documents=True
    )

    # 5. 질문에 대한 답변 생성
    result = qa_chain.run(question)

    return result
