import os
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from google.colab import drive
import evaluate

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Mount Google Drive (if necessary)
drive.mount('/content/drive')

# Step 1: Load the dataset
df = pd.read_excel("/content/drive/My Drive/Thesis_Dataset/subtitle.xlsx")  # Adjust path

# Ensure the dataset has two columns: 'text' and 'summary'
df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "summary"})
df = df.dropna()

# Step 2: Split the dataset (80% for training, 20% for validation)
train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)

# Convert pandas DataFrames into Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Combine datasets into a DatasetDict for Hugging Face Trainer
dataset_dict = DatasetDict({"train": train_dataset, "val": val_dataset})

# Step 3: Load the tokenizer and model (using Bangla T5 for summarization)
model_name = "csebuetnlp/banglat5"  # Ensure this is a seq2seq model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenization function for summarization
def preprocess_function(examples):
    inputs = tokenizer(examples['text'], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples['summary'], max_length=150, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    inputs["labels"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_ids]
        for label_ids in inputs["labels"]
    ]
    return inputs

# Apply tokenization to the train and validation datasets
tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

# Step 4: Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    predict_with_generate=True,
    run_name="bangla_summarization_run"  # Optional: set a different run name
)

# Step 5: Define a trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    tokenizer=tokenizer,
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the fine-tuned model and tokenizer
def save_model(model, output_dir):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    model.save_pretrained(output_dir)

save_model(model, "/content/drive/My Drive/Thesis_Dataset/fine_tuned_bangla_bert")
tokenizer.save_pretrained("/content/drive/My Drive/Thesis_Dataset/fine_tuned_bangla_bert")

# Step 8: Generate a summary using the fine-tuned model
def generate_summary(text):
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors="pt", padding="max_length")
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_beams=3,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Generate summaries for the test dataset
val_df['generated_summary'] = val_df['text'].apply(generate_summary)

# Step 9: Load the ROUGE metric from the 'evaluate' library
rouge = evaluate.load('rouge')

# Step 10: Calculate the final ROUGE score for all predictions and references

# Collect all references and generated summaries
references = val_df['summary'].tolist()
predictions = val_df['generated_summary'].tolist()

# Compute the final ROUGE score across the whole validation set
final_rouge_score = rouge.compute(predictions=predictions, references=references)

# Print the final ROUGE scores
print(f"Final ROUGE Score: {final_rouge_score}")

# Save the output to a new Excel file with generated summaries and final ROUGE scores
val_df.to_excel("/content/drive/My Drive/Thesis_Dataset/test_data_with_summaries_and_rouge.xlsx", index=False)

# Print the result with generated summaries
print(val_df[['text', 'summary', 'generated_summary']])

