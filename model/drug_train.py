from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# 1. DrugBank 데이터셋 로드
dataset = load_dataset("SkyHuReal/DrugBank-Alpaca")

# 데이터셋 확인 (훈련 및 검증 세트)
print(dataset)

# 2. BART 모델 및 토크나이저 로드
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 3. 데이터 전처리 함수 정의
def preprocess(examples):

    inputs = tokenizer(examples['input'], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

# 4. 데이터셋 전처리
tokenized_dataset = dataset.map(preprocess, batched=True)

# 5. 훈련 인수 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./drug_recommender_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=4,
    save_total_limit=2,
    logging_dir="./logs",  # 로그 파일 저장 디렉토리 설정
    logging_steps=10,      # 10 스텝마다 로그 기록
)

# 6. Seq2SeqTrainer 정의
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # 훈련 데이터셋
    tokenizer=tokenizer,
)

# 7. 훈련 시작
trainer.train()