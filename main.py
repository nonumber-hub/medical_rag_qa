import os
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config.settings import INDEX_PATH, EMBEDDING_MODEL_NAME, MODEL_PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_vectorstore():
    embeddings = load_embeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

def load_generator_all():
    models = {}
    for model_type, model_path in MODEL_PATHS.items():
        abs_path = os.path.abspath(model_path).replace("\\", "/")
        print(f"[Debug] Loading model '{model_type}' from: {abs_path}")
        tokenizer = AutoTokenizer.from_pretrained(abs_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(abs_path, local_files_only=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        models[model_type] = HuggingFacePipeline(pipeline=pipe)
    return models

def classify_question(question: str) -> str:
    question = question.lower()

    disease_keywords = ["fever", "cough", "infection", "symptom", "pain", "flu", "disease"]
    drug_keywords = ["medicine", "drug", "prescription", "dose", "treatment", "side effect"]
    
    if any(keyword in question for keyword in disease_keywords):
        return "disease"
    elif any(keyword in question for keyword in drug_keywords):
        return "drug"
    elif question.strip() == "":
        return "default"
    else:
        return "qa"

def main():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = input("What can I help you with?: ")
    model_type = classify_question(question)

    # ✅ 모델 전체 로드 후 선택
    generators = load_generator_all()
    generator = generators[model_type]

    # 🔍 모델 단독 출력 테스트
    print("\n[Test] 모델 단독 출력:")
    single_output = generator(question)

    # 디버깅: 출력 타입 확인
    print(f"[Debug] generator output type: {type(single_output)}")
    print(f"[Debug] generator output raw: {single_output}")

    # 결과 안전하게 출력
    if isinstance(single_output, str):
        print(single_output)
    elif isinstance(single_output, list) and isinstance(single_output[0], dict) and "generated_text" in single_output[0]:
        print(single_output[0]["generated_text"])
    else:
        print("[Error] Unexpected output format from generator")

    print("\n----------\n")

    # 📚 RetrievalQA 체인 실행
    qa_chain = RetrievalQA.from_chain_type(
        llm=generator,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})

    print("\n[Answer]")
    print(result['result'])

    print("\n[Source Documents]")
    for doc in result['source_documents']:
        print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
