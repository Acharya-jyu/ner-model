# Medical Entity Recognition Model using BioMedBERT
## Overview
This repository contains the implementation of a specialized Medical Entity Recognition system using transfer learning with BiomedNLP-PubMedBERT. The model achieves state-of-the-art performance in identifying medical entities with an F1 score of 0.9868 for strict entity matching.
Key Features

## Fine-tuned BiomedNLP-BioMedBERT model for medical entity recognition
<li> Comprehensive data preprocessing pipeline</li>
<li> Advanced entity refinement with subword merging</li>
 <li>Sophisticated error analysis and performance metrics</li>
 <li>Support for complex medical terminology </li>

# Performance Metrics

## Strict Entity Matching:

<li>Precision: 0.9869</li>
<li>Recall: 0.9868</li>
<li>F1 Score: 0.9868</li>

## Partial Entity Matching:

<li>Precision: 0.9527</li>
<li>Recall: 0.9456</li>
<li>F1 Score: 0.9491</li>



## Repository Structure
<pre>├── data_loader.py        # Dataset loading and preparation
├── model_trainer.py      # Model training implementation
├── model_tester.py       # Testing and evaluation scripts 
├── requirements.txt      # Project dependencies 
└── README.md             # Project documentation </pre>

# Installation

## Clone the repository:

git clone https://github.com/yourusername/medical-entity-recognition.git
cd medical-entity-recognition

## Install dependencies:

pip install -r requirements.txt

## Dataset
The model is trained on the biomedical NER dataset comprising 21,225 annotated medical sentences:

<li>Training set: 15,488 sentences (73%)</li>
<li>Testing set: 5,737 sentences (27%)</li>
<li>Implementation of BIO (Beginning, Inside, Outside) tagging scheme</li>

## Usage
Data Preparation
python data_loader.py
Model Training
python model_trainer.py
Testing the Model
python model_tester.py


## Model Architecture

<b>Base Model:</b> microsoft/BiomedNLP-BioMedBERT-base-uncased-abstract-fulltext<br/>
- Specialized configuration for medical entity recognition
- Implementation of advanced tokenization and label alignment
- Comprehensive error handling and boundary refinement

## Training Configuration

<ul>Learning Rate: 5e-5</ul>
<ul>Batch Size: 8</ul>
<ul>Training Epochs: 15</ul>
<ul>Weight Decay: 0.01</ul>
<ul>Warmup Steps: 500</ul>
<ul>Early Stopping: Patience of 5 epochs</ul>

## Citation (bibtex)
@article{medical-ner-2025,
  title={Medical Entity Recognition System Using Transfer Learning},
  author={Aashish Acharya},
  year={2025}
}
## Pretrained Model
The fine-tuned model is available on Hugging Face: acharya-jyu/BioMed-NER-English
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- Microsoft Research for the BiomedNLP-PubMedBERT base model
- The biomedical NER dataset contributors in Hugging Face (rjac/biobert-ner-diseases-dataset)
- Hugging Face team for their transformer library
