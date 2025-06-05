from datasets import load_dataset
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from transformers import default_data_collator

# 1. 데이터셋 로드 (SQuAD v2)
dataset = load_dataset("rajpurkar/squad_v2")

# 2. BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased")

# 3. 전처리 함수
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    
    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"])):
        if len(examples["answers"][i]["text"]) == 0:
            # 답이 없는 경우 → BERT에선 start/end를 0으로 설정 (또는 -100)
            start_positions.append(0)
            end_positions.append(0)
        else:
            start = examples["answers"][i]["answer_start"][0]
            end = start + len(examples["answers"][i]["text"][0])
            start_positions.append(start)
            end_positions.append(end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

# 4. 데이터셋 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. 훈련 설정
training_args = TrainingArguments(
    output_dir="./qa_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    save_steps=10000,
    logging_dir="./logs",
    logging_steps=10000,
    save_total_limit=2,
    fp16=True,
)

# 6. Trainer 객체 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

# 7. 훈련
trainer.train()
