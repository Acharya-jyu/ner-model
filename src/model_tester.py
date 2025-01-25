from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the Fine-Tuned Model and Tokenizer
model_path = "/content/drive/My Drive/4PubMedBert-BioBERTNER-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Define Label Mapping
id_to_label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

# Cleaning Function for Subword Artifacts and Possessives
def clean_entity(entity):
    entity = entity.replace("##", "").strip()  # Remove subword artifacts
    entity = entity.replace(" ' ", "'").replace("  ", " ")  # Fix possessives
    entity = entity.replace(" s ", "'s ").replace("rash es", "rashes")  # Additional fixes
    return entity.capitalize()

# Subword Merging
def merge_subwords(tokens, tags):
    entities = []
    current_entity = []

    for token, tag in zip(tokens, tags):
        if tag in ["B-Disease", "I-Disease"]:
            current_entity.append(token)
        else:
            if current_entity:
                entity = " ".join(current_entity).replace("##", "").strip()
                entities.append(clean_entity(entity))
                current_entity = []
    if current_entity:
        entity = " ".join(current_entity).replace("##", "").strip()
        entities.append(clean_entity(entity))
    return entities

# Deduplicate Entities
def filter_duplicates(entities):
    return list(set(entities))

# Split Listed Entities
def split_listed_entities(entities):
    split_entities = []
    for entity in entities:
        if ',' in entity:
            split_entities.extend([e.strip() for e in entity.split(',')])
        else:
            split_entities.append(entity)
    return split_entities

# Full Refinement Pipeline
def refine_entities(entities):
    # Deduplicate entities
    entities = filter_duplicates(entities)
    # Split listed entities
    entities = split_listed_entities(entities)
    # Clean each entity
    return [clean_entity(entity) for entity in entities]

# Prediction Function with Full Refinement
def predict_ner_with_refinements(text):
    # Tokenize Input Text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**tokens)

    # Get Predictions
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze())
    tags = [id_to_label[pred] for pred in predictions]

    # Debugging: Print Tokens and Tags
    print("\nTokens and Tags:")
    for token, tag in zip(tokens_decoded, tags):
        print(f"{token}: {tag}")

    # Merge Subwords
    merged_entities = merge_subwords(tokens_decoded, tags)
    # Apply Refinements
    refined_entities = refine_entities(merged_entities)

    return refined_entities

# Test Sentences with Multi-Line Inputs and Complex Cases
test_sentences = [
    "The patient was diagnosed with diabetes and hypertension.\nHe also reported occasional chest pain and fatigue.",
    "Symptoms include fever, cough, difficulty breathing, and weight loss.\nBlood tests revealed anemia and elevated white blood cell counts.",
    "Severe muscle pain, joint inflammation, and skin rashes were observed.\nThese could indicate lupus or another autoimmune disease.",
    "The patient has a history of heart failure and atrial fibrillation.\nCurrent symptoms include shortness of breath and dizziness.",
    "Chronic obstructive pulmonary disease (COPD) was diagnosed along with asthma.\nPatient has persistent coughing and difficulty sleeping.",
    "A biopsy revealed lymphoma and leukemia.\nThe patient is undergoing chemotherapy for both conditions.",
    "Early-onset Alzheimer's disease was noted along with mild cognitive impairment.\nNo other neurological symptoms were observed.",
]

# Run Predictions
print("Testing Model with Full Refinements and Multi-Line Inputs...\n")
for i, sentence in enumerate(test_sentences, 1):
    print(f"Test Sentence {i}: {sentence}")
    predictions = predict_ner_with_refinements(sentence)
    print("Refined Extracted Entities:", ", ".join(predictions) if predictions else "No entities detected.")
    print("\n")
