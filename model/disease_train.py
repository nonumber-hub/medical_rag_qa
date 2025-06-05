from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# 1. 데이터셋 로드
dataset = load_dataset("leowei31/mimic_note_summarization")

# 2. BART 모델과 토크나이저 불러오기
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 3. 전처리 함수 정의
def preprocess_function(examples):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# 4. 데이터셋 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# 5. 훈련 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_mimic_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=True,  # CUDA 가속 사용 시
    logging_steps=50,
    save_steps=500
)

# 6. Trainer 정의
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer
)

# 7. 훈련 시작
trainer.train()