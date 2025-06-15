from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "my-emotion-model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

res = classifier("input text")
print(res)