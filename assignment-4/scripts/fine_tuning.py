import os
import pickle
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
import optuna
import json
from sklearn.metrics import f1_score
import torch
from torch import nn
import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4" 

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    roberta_trial_paths = glob.glob("./results/roberta_trial_*")
    distilbert_trial_paths = glob.glob("./results/distilbert_trial_*")
    
    roberta_completed_trials = len(roberta_trial_paths)
    distilbert_completed_trials = len(distilbert_trial_paths)
    
    print(f"Found {roberta_completed_trials} completed RoBERTa trials")
    print(f"Found {distilbert_completed_trials} completed DistilBERT trials")
    
    dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
    num_labels = len(dataset["train"].features["labels"].feature.names)
    
    roberta_best_params = {}
    if roberta_completed_trials > 0:
        roberta_best_params = {
            "learning_rate": 2e-5, 
            "weight_decay": 0.05, 
            "batch_size": 8,       
        }
        print(f"Using existing best params for RoBERTa: {roberta_best_params}")
    
    max_length = 40  
    cache_dir = "./model_cache" 
    os.makedirs(cache_dir, exist_ok=True)
    
    roberta_dataset_path = "./preprocessed_roberta_data.pkl"
    distilbert_dataset_path = "./preprocessed_distilbert_data.pkl"
    
    if os.path.exists(roberta_dataset_path) and os.path.exists(distilbert_dataset_path):
        print("Loading preprocessed datasets...")
        with open(roberta_dataset_path, "rb") as f:
            roberta_encoded_dataset = pickle.load(f)
        with open(distilbert_dataset_path, "rb") as f:
            distilbert_encoded_dataset = pickle.load(f)
        
        roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", cache_dir=cache_dir)
        distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
    else:
        print("Preprocessing datasets...")
        roberta_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", cache_dir=cache_dir)
        distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir=cache_dir)
        
        def preprocess_function(examples, tokenizer):
            tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)
            tokenized["labels"] = [
                [1.0 if i in label else 0.0 for i in range(num_labels)]
                for label in examples["labels"]
            ]
            return tokenized

        batch_size = 64  
        
        roberta_encoded_dataset = dataset.map(
            lambda examples: preprocess_function(examples, roberta_tokenizer),
            batched=True,
            batch_size=batch_size,
            remove_columns=["text", "id"],
            num_proc=2  
        )
        
        distilbert_encoded_dataset = dataset.map(
            lambda examples: preprocess_function(examples, distilbert_tokenizer),
            batched=True,
            batch_size=batch_size,
            remove_columns=["text", "id"],
            num_proc=2 
        )
        
        with open(roberta_dataset_path, "wb") as f:
            pickle.dump(roberta_encoded_dataset, f)
        with open(distilbert_dataset_path, "wb") as f:
            pickle.dump(distilbert_encoded_dataset, f)
    
    roberta_data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    distilbert_data_collator = DataCollatorWithPadding(tokenizer=distilbert_tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits))
        predictions = (probs > 0.5).int()
        return {
            "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
        }

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return (loss, outputs) if return_outputs else loss

    def objective_roberta(trial):
        if roberta_best_params:
            learning_rate = trial.suggest_float("learning_rate", 
                                               roberta_best_params["learning_rate"] * 0.5, 
                                               roberta_best_params["learning_rate"] * 1.5, 
                                               log=True)
            weight_decay = trial.suggest_float("weight_decay", 
                                              roberta_best_params["weight_decay"] * 0.5, 
                                              roberta_best_params["weight_decay"] * 1.5)
            batch_size = trial.suggest_categorical("batch_size", [4, 8])  
        else:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
            batch_size = trial.suggest_categorical("batch_size", [4, 8])  

        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-base",
            num_labels=num_labels,
            problem_type="multi_label_classification",
            cache_dir=cache_dir
        )

        model.gradient_checkpointing_enable()
   
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=f"./results/roberta_trial_{trial.number}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2, 
            num_train_epochs=3,
            weight_decay=weight_decay,
            eval_strategy="steps",
            eval_steps=100,        
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            push_to_hub=False,
            dataloader_num_workers=2, 
            gradient_accumulation_steps=2, 
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            logging_steps=50,
            save_total_limit=2, 
        )

        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=roberta_encoded_dataset["train"],
            eval_dataset=roberta_encoded_dataset["validation"],
            tokenizer=roberta_tokenizer,
            data_collator=roberta_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        eval_result = trainer.evaluate()
        return eval_result["eval_f1"]

    def objective_distilbert(trial):
        if roberta_best_params:
            learning_rate = trial.suggest_float("learning_rate", 
                                              roberta_best_params["learning_rate"] * 0.5, 
                                              roberta_best_params["learning_rate"] * 1.5, 
                                              log=True)
            weight_decay = trial.suggest_float("weight_decay", 
                                             roberta_best_params["weight_decay"] * 0.5, 
                                             roberta_best_params["weight_decay"] * 1.5)
            batch_size = trial.suggest_categorical("batch_size", [4, 8])  
        else:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
            batch_size = trial.suggest_categorical("batch_size", [4, 8]) 

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            problem_type="multi_label_classification",
            cache_dir=cache_dir
        )
        
        model.gradient_checkpointing_enable()
        
        model = model.to(device)

        training_args = TrainingArguments(
            output_dir=f"./results/distilbert_trial_{trial.number}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=3,
            weight_decay=weight_decay,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            push_to_hub=False,
            dataloader_num_workers=2,
            gradient_accumulation_steps=2,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            logging_steps=50,
            save_total_limit=2,
        )

        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=distilbert_encoded_dataset["train"],
            eval_dataset=distilbert_encoded_dataset["validation"],
            tokenizer=distilbert_tokenizer,
            data_collator=distilbert_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()
        eval_result = trainer.evaluate()
        return eval_result["eval_f1"]

    pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=100)
    
    if roberta_completed_trials == 0:
        print("Optimizing RoBERTa hyperparameters...")
        study_roberta = optuna.create_study(direction="maximize", pruner=pruner)
        study_roberta.optimize(objective_roberta, n_trials=2)  
        roberta_best_params = study_roberta.best_params
    else:
        study_roberta = None
        print("Skipping RoBERTa hyperparameter optimization (using existing results)")
    
    print("Optimizing DistilBERT hyperparameters...")
    study_distilbert = optuna.create_study(direction="maximize", pruner=pruner)
    study_distilbert.optimize(objective_distilbert, n_trials=2) 

    if study_roberta:
        print("\nBest hyperparameters for RoBERTa:", study_roberta.best_params)
    else:
        print("\nUsing predefined hyperparameters for RoBERTa:", roberta_best_params)
    print("Best hyperparameters for DistilBERT:", study_distilbert.best_params)

    print("\nTraining final RoBERTa model...")
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base",
        num_labels=num_labels,
        problem_type="multi_label_classification",
        cache_dir=cache_dir
    )
    roberta_model.gradient_checkpointing_enable()
    roberta_model = roberta_model.to(device)
    
    best_roberta_params = study_roberta.best_params if study_roberta else roberta_best_params
    
    roberta_training_args = TrainingArguments(
        output_dir="./results/roberta_emotions_final",
        learning_rate=best_roberta_params["learning_rate"],
        per_device_train_batch_size=best_roberta_params["batch_size"],
        per_device_eval_batch_size=best_roberta_params["batch_size"] * 2,
        num_train_epochs=5,
        weight_decay=best_roberta_params["weight_decay"],
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        push_to_hub=False,
        dataloader_num_workers=2,
        gradient_accumulation_steps=4,  
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=50,
        save_total_limit=2,
    )

    roberta_trainer = MultiLabelTrainer(
        model=roberta_model,
        args=roberta_training_args,
        train_dataset=roberta_encoded_dataset["train"],
        eval_dataset=roberta_encoded_dataset["validation"],
        tokenizer=roberta_tokenizer,
        data_collator=roberta_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    roberta_trainer.train()

    print("\nTraining final DistilBERT model...")
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification",
        cache_dir=cache_dir
    )
    distilbert_model.gradient_checkpointing_enable()
    distilbert_model = distilbert_model.to(device)

    distilbert_training_args = TrainingArguments(
        output_dir="./results/distilbert_emotions_final",
        learning_rate=study_distilbert.best_params["learning_rate"],
        per_device_train_batch_size=study_distilbert.best_params["batch_size"],
        per_device_eval_batch_size=study_distilbert.best_params["batch_size"] * 2,
        num_train_epochs=5,
        weight_decay=study_distilbert.best_params["weight_decay"],
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        push_to_hub=False,
        dataloader_num_workers=2,
        gradient_accumulation_steps=4,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=50,
        save_total_limit=2,
    )

    distilbert_trainer = MultiLabelTrainer(
        model=distilbert_model,
        args=distilbert_training_args,
        train_dataset=distilbert_encoded_dataset["train"],
        eval_dataset=distilbert_encoded_dataset["validation"],
        tokenizer=distilbert_tokenizer,
        data_collator=distilbert_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    distilbert_trainer.train()

    print("\nEvaluating models on test set...")
    roberta_results = roberta_trainer.evaluate(roberta_encoded_dataset["test"])
    distilbert_results = distilbert_trainer.evaluate(distilbert_encoded_dataset["test"])

    print(f"\nRoBERTa Test Results: {roberta_results}")
    print(f"DistilBERT Test Results: {distilbert_results}")

    roberta_trainer.save_model("./roberta_emotions_final")
    distilbert_trainer.save_model("./distilbert_emotions_final")

    with open("fine_tuning_results.json", "w") as f:
        json.dump({
            "roberta": {
                "best_params": best_roberta_params,
                "test_results": roberta_results
            },
            "distilbert": {
                "best_params": study_distilbert.best_params,
                "test_results": distilbert_results
            }
        }, f, indent=4)

    print("\nTraining complete! Models and results saved.")

if __name__ == '__main__':
    main()