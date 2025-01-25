# Fine Tune the model
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
from google.colab import drive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers.trainer_callback import EarlyStoppingCallback
import json

# 1. Load Dataset
dataset_path = "/content/drive/My Drive/4PubMedBert-BioBERTNER"
dataset = load_from_disk(dataset_path)

# 2. Define Label Mapping
label_list = ["O", "B-Disease", "I-Disease"]
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# 3. Load Tokenizer and Model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id)

# 4. Preprocessing Function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their corresponding word
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # Label for the first subtoken
            else:
                label_ids.append(-100)  # Assign -100 to subtokens
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply preprocessing to dataset
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# 5. Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

def plot_confusion_matrix(labels, predictions, output_path):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()

# 6. Compute Metrics
def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = np.argmax(pred.predictions, axis=2).flatten()

    # Remove ignored indices
    true_labels = labels[labels != -100]
    true_preds = preds[labels != -100]

    # 1. Strict F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_preds, average="weighted"
    )

    # 2. Partial Match F1
    true_entities = get_entity_spans(true_labels)
    pred_entities = get_entity_spans(true_preds)
    partial_matches = get_partial_matches(true_entities, pred_entities)

    partial_precision = partial_matches / max(len(pred_entities), 1)
    partial_recall = partial_matches / max(len(true_entities), 1)
    partial_f1 = 2 * (partial_precision * partial_recall) / max(partial_precision + partial_recall, 1e-10)

    # 3. Error Distribution
    error_dist = get_error_distribution(
        [id_to_label[l] for l in true_labels],
        [id_to_label[p] for p in true_preds]
    )

    # Generate confusion matrix
    plot_confusion_matrix(
        true_labels,
        true_preds,
        f"confusion_matrix.png"
    )

     # Print metrics in a readable format
    print("\n" + "="*50)
    print("TRAINING METRICS:")
    print("="*50)

    print("\n1. Strict Entity Matching:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\n2. Partial Entity Matching:")
    print(f"Precision: {partial_precision:.4f}")
    print(f"Recall: {partial_recall:.4f}")
    print(f"F1 Score: {partial_f1:.4f}")

    print("\n3. Error Analysis:")
    print(f"Boundary Errors: {error_dist['boundary_errors']}")
    print(f"Type Errors: {error_dist['type_errors']}")
    print("="*50 + "\n")

    return {
        "strict_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "partial_metrics": {
            "precision": partial_precision,
            "recall": partial_recall,
            "f1": partial_f1
        },
        "error_analysis": error_dist,
        "f1": f1
    }
def get_partial_matches(true_ents, pred_ents):
    matches = 0
    for t_start, t_end in true_ents:
        for p_start, p_end in pred_ents:
            if (t_start <= p_end and p_start <= t_end):
                matches += 1
                break
    return matches
def get_error_distribution(true_labels, pred_labels):
    boundary_errors = 0
    type_errors = 0
    for t, p in zip(true_labels, pred_labels):
        if t.startswith('B-') and p.startswith('B-'):
            if t != p:
                type_errors += 1
        elif t.startswith('B-') or p.startswith('B-'):
            boundary_errors += 1
    return {'boundary_errors': boundary_errors, 'type_errors': type_errors}

def get_entity_spans(labels):
    spans = set()
    start = None
    for i, label in enumerate(labels):
        if label == 1:  # B-Disease
            if start is not None:
                spans.add((start, i))
            start = i
        elif label == 0:  # O
            if start is not None:
                spans.add((start, i))
                start = None
    if start is not None:
        spans.add((start, len(labels)))
    return spans

# 7. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=5,
    logging_dir="./logs",
    logging_steps=50,
    # Add these configurations
    warmup_steps=500,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5,
               early_stopping_threshold=0.00001)]
)


# 9. Train the Model
trainer.train()


# 10. Save the Model
model_save_path = "/content/drive/My Drive/4PubMedBert-BioBERTNER/PubMedBert-BioBERTNER-finetuned"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# After training
eval_results = trainer.evaluate()
eval_output_path = os.path.join(model_save_path, "eval_results.json")
with open(eval_output_path, "a") as f:
    json.dump(eval_results, f)

print(f"Model saved to: {model_save_path}")
