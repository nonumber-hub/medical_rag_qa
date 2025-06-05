# rag_pipeline.py
from rag_chain import build_rag_chain

def run_rag():
    rag_chain = build_rag_chain()
    print("RAG 시스템이 초기화되었습니다. 질문을 입력하세요.")

    while True:
        question = input("\n질문: ")
        if question.lower() in ["exit", "quit"]:
            break

        result = rag_chain.run(question)
        print(f"📘 응답: {result}")

if __name__ == "__main__":
    run_rag()
