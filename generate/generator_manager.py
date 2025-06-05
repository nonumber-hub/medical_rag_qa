from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config.settings import BART_MODEL_PATH

class GeneratorManager:
    """
    제너레이터 모델을 관리하는 클래스입니다.
    """
    def __init__(self):
        # 모델 이름과 경로 설정
        self.model_paths = {
            "disease": "path/to/disease_model",
            "drug": "path/to/drug_model",
            "qa": "path/to/qa_model"
        }
        self.generators = {}

def load_generator(self, model_type: str):
    """
    주어진 모델 유형에 맞는 BART 모델을 로드합니다.
    """
    if model_type not in self.model_paths:
        raise ValueError(f"모델 유형이 잘못되었습니다: {model_type}")

    if model_type not in self.generators:
        model_path = self.model_paths[model_type]

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0,  # CUDA 사용
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

        self.generators[model_type] = pipe

    return self.generators[model_type]

def get_model(self, model_type: str):
    """
    이미 로드된 제너레이터를 반환합니다.
    
    Args:
    model_type (str): 'disease', 'drug', 또는 'qa'
    
    Returns:
    generator: 로드된 모델
    """
    if model_type not in self.generators:
        return self.load_generator(model_type)
    return self.generators[model_type]

def list_available_models(self):
    """
    로드 가능한 모델 목록을 반환합니다.
    
    Returns:
        list: 모델 이름 리스트
    """
    return list(self.model_paths.keys())

