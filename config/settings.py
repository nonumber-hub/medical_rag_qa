import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DOCUMENT_DIR = os.path.join("data", "documents")
print(f"[✓] 문서 디렉토리 경로: {DOCUMENT_DIR}")

if not os.path.exists(DOCUMENT_DIR):
    print(f"[✗] 디렉토리 존재하지 않음: {DOCUMENT_DIR}")
else:
    print(f"[✓] 디렉토리가 존재합니다.")
    for filename in os.listdir(DOCUMENT_DIR):
        print(f"[✓] 파일: {filename}")

INDEX_PATH = os.path.join("data", "faiss_index")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

MODEL_PATHS = {
    "disease": os.path.join("path", "disease_model"),
    "drug": os.path.join("path", "drug_model"),
    "qa": os.path.join("path", "qa_model"),
    "default": os.path.join("path", "qa_model"),
}

LOG_DIR = "logs"

# 상대경로 기준 폴더 생성
os.makedirs(DOCUMENT_DIR, exist_ok=True)
os.makedirs(INDEX_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
