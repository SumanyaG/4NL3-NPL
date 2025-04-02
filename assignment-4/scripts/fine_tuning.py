import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import optuna
from functools import partial
import json

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
num_labels = len(dataset["train"].features["labels"].feature.names)

roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to datasets
roberta_encoded_dataset = dataset.map(
    lambda examples: preprocess_function(examples, roberta_tokenizer), 
    batched=True
)
distilbert_encoded_dataset = dataset.map(
    lambda examples: preprocess_function(examples, distilbert_tokenizer), 
    batched=True
)

for encoded_dataset in [roberta_encoded_dataset, distilbert_encoded_dataset]:
    encoded_dataset = encoded_dataset.remove_columns(["text", "id"]).rename_column("labels", "label")
    encoded_dataset.set_format("torch")

metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

def objective_roberta(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16]) 
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", 
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir=f"./results/roberta_trial_{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,  # Limited epochs to prevent overfitting and save compute
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,  
        dataloader_num_workers=2,  
        gradient_accumulation_steps=2, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=roberta_encoded_dataset["train"],
        eval_dataset=roberta_encoded_dataset["validation"],
        tokenizer=roberta_tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )
    
    trainer.train()
    eval_result = trainer.evaluate(roberta_encoded_dataset["validation"])
    
    return eval_result["eval_f1"] 

def objective_distilbert(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])  
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir=f"./results/distilbert_trial_{trial.number}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,
        dataloader_num_workers=2,
        gradient_accumulation_steps=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=distilbert_encoded_dataset["train"],
        eval_dataset=distilbert_encoded_dataset["validation"],
        tokenizer=distilbert_tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    eval_result = trainer.evaluate(distilbert_encoded_dataset["validation"])
    
    return eval_result["eval_f1"]

study_roberta = optuna.create_study(direction="maximize")
study_roberta.optimize(objective_roberta, n_trials=5) 

study_distilbert = optuna.create_study(direction="maximize")
study_distilbert.optimize(objective_distilbert, n_trials=5)

print("Best hyperparameters for RoBERTa:", study_roberta.best_params)
print("Best hyperparameters for DistilBERT:", study_distilbert.best_params)

roberta_model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base", 
    num_labels=num_labels
)

roberta_training_args = TrainingArguments(
    output_dir="./results/roberta_emotions_final",
    learning_rate=study_roberta.best_params["learning_rate"],
    per_device_train_batch_size=study_roberta.best_params["batch_size"],
    per_device_eval_batch_size=study_roberta.best_params["batch_size"],
    num_train_epochs=5,
    weight_decay=study_roberta.best_params["weight_decay"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,
    dataloader_num_workers=2,
    gradient_accumulation_steps=2,
)

roberta_trainer = Trainer(
    model=roberta_model,
    args=roberta_training_args,
    train_dataset=roberta_encoded_dataset["train"],
    eval_dataset=roberta_encoded_dataset["validation"],
    tokenizer=roberta_tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Training final RoBERTa model...")
roberta_trainer.train()

distilbert_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=num_labels
)

distilbert_training_args = TrainingArguments(
    output_dir="./results/distilbert_emotions_final",
    learning_rate=study_distilbert.best_params["learning_rate"],
    per_device_train_batch_size=study_distilbert.best_params["batch_size"],
    per_device_eval_batch_size=study_distilbert.best_params["batch_size"],
    num_train_epochs=5,
    weight_decay=study_distilbert.best_params["weight_decay"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,
    dataloader_num_workers=2,
    gradient_accumulation_steps=2,
)

distilbert_trainer = Trainer(
    model=distilbert_model,
    args=distilbert_training_args,
    train_dataset=distilbert_encoded_dataset["train"],
    eval_dataset=distilbert_encoded_dataset["validation"],
    tokenizer=distilbert_tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Training final DistilBERT model...")
distilbert_trainer.train()

roberta_results = roberta_trainer.evaluate(roberta_encoded_dataset["test"])
distilbert_results = distilbert_trainer.evaluate(distilbert_encoded_dataset["test"])

print(f"RoBERTa Test Results: {roberta_results}")
print(f"DistilBERT Test Results: {distilbert_results}")

roberta_trainer.save_model("./roberta_emotions_final")
distilbert_trainer.save_model("./distilbert_emotions_final")

with open("fine_tuning_results.json", "w") as f:
    json.dump({
        "roberta": {
            "best_params": study_roberta.best_params,
            "test_results": roberta_results
        },
        "distilbert": {
            "best_params": study_distilbert.best_params,
            "test_results": distilbert_results
        }
    }, f)