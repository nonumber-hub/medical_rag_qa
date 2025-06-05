from vllm import LLM, SamplingParams

# 로컬에서 LLaMA 모델 서빙
llm = LLM(model="/path/to/llama-2-7b-chat")

# 샘플링 파라미터 설정
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

# 질의 처리 함수
def ask(question: str):
    outputs = llm.generate(prompt=question, sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()

if __name__ == "__main__":
    while True:
        q = input("Q: ")
        print("A:", ask(q))
