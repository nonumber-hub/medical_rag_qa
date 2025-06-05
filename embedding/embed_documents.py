import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document  # 핵심 Document 클래스 위치
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import DOCUMENT_DIR, INDEX_PATH, EMBEDDING_MODEL_NAME


def load_csv_documents():
    """
    CSV 파일을 로드하고 각 행을 문장으로 구성한 Document 리스트로 반환합니다.
    """
    documents = []
    for filename in os.listdir(DOCUMENT_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DOCUMENT_DIR, filename)
            try:
                df = pd.read_csv(file_path)

                # 컬럼 헤더 공백 제거 (중요!)
                df.columns = df.columns.str.strip()

                for _, row in df.iterrows():
                    try:
                        text = (
                            f"Disease: {row.get('Disease', 'Unknown')}, "
                            f"Fever: {row.get('Fever', 'Unknown')}, "
                            f"Cough: {row.get('Cough', 'Unknown')}, "
                            f"Fatigue: {row.get('Fatigue', 'Unknown')}, "
                            f"Difficulty Breathing: {row.get('Difficulty Breathing', 'Unknown')}, "
                            f"Age: {row.get('Age', 'Unknown')}, "
                            f"Gender: {row.get('Gender', 'Unknown')}, "
                            f"Blood Pressure: {row.get('Blood Pressure', 'Unknown')}, "
                            f"Cholesterol Level: {row.get('Cholesterol Level', 'Unknown')}, "
                            f"Outcome: {row.get('Outcome Variable', 'Unknown')}"
                        )
                        documents.append(Document(page_content=text, metadata={"source": filename}))
                    except Exception as row_e:
                        print(f"[!] 행 처리 오류 (건너뜀): {row_e}")
            except Exception as e:
                print(f"[✗] CSV 로딩 오류: {file_path}, 오류: {e}")
    print(f"[✓] CSV에서 로드된 문서 수: {len(documents)}")
    return documents


def build_index():
    """
    문서를 불러와 임베딩을 생성하고 FAISS 인덱스를 저장합니다.
    """
    all_docs = []

    # 1. 텍스트 파일 로드
    for filename in os.listdir(DOCUMENT_DIR):
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(DOCUMENT_DIR, filename), encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
                print(f"[✓] {filename} 로드 완료: {len(docs)}개 문서")
            except Exception as e:
                print(f"[✗] 텍스트 파일 로딩 오류: {filename}, 오류: {e}")

    # 2. CSV 문서 로드
    csv_docs = load_csv_documents()
    all_docs.extend(csv_docs)

    if not all_docs:
        print("[✗] 문서가 없습니다. 데이터를 다시 확인하세요.")
        return

    # 3. 문서 분할
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )
        split_docs = splitter.split_documents(all_docs)
        print(f"[✓] 문서 분할 완료: {len(split_docs)}개의 청크.")
    except Exception as e:
        print(f"[✗] 문서 분할 오류: {e}")
        return

    # 4. 임베딩 생성
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print(f"[✓] 임베딩 모델 로드 완료: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"[✗] 임베딩 모델 로드 오류: {e}")
        return

    # 5. FAISS 벡터 저장소 생성
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        print(f"[✓] FAISS 벡터 저장소 생성 완료.")
    except Exception as e:
        print(f"[✗] FAISS 벡터 저장소 생성 오류: {e}")
        return

    # 6. 인덱스 저장
    try:
        vectorstore.save_local(INDEX_PATH)
        print(f"[✓] 벡터 저장소 저장 완료: {INDEX_PATH}")
    except Exception as e:
        print(f"[✗] 벡터 저장소 저장 오류: {e}")


if __name__ == "__main__":
    build_index()
