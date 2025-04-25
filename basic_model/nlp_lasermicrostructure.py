import streamlit as st
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="PDF Token Generator & LLM Fine-Tuner", layout="wide")

# Title and description
st.title("PDF Token Generator and LLM Fine-Tuner")
st.markdown("""
Upload a PDF file to extract text, generate token embeddings, and fine-tune a DistilBERT model.
The app extracts text using PyPDF2, generates embeddings with SentenceTransformers, and fine-tunes
DistilBERT for a sample text classification task.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file consisting of the study about laser and/or microstructure", type="pdf")

def extract_text_from_pdf(file):
    """Extract text from a PDF file using PyPDF2."""
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Read PDF
        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return text if text.strip() else "No text extracted from the PDF."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def generate_embeddings(text):
    """Generate token embeddings using SentenceTransformers."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Split text into sentences for embedding
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        embeddings = model.encode(sentences, show_progress_bar=True)
        return sentences, embeddings
    except Exception as e:
        return [], f"Error generating embeddings: {str(e)}"

def prepare_dataset(text):
    """Prepare a dataset for fine-tuning (dummy classification task)."""
    # Split text into sentences
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    
    # Create dummy labels (e.g., 1 for positive, 0 for negative) for demonstration
    # In a real scenario, you'd need labeled data or a specific task
    labels = [1 if i % 2 == 0 else 0 for i in range(len(sentences))]
    
    # Tokenize sentences
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128)
    
    # Create dataset
    dataset_dict = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }
    return Dataset.from_dict(dataset_dict)

def fine_tune_model(dataset):
    """Fine-tune DistilBERT on the provided dataset."""
    try:
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,  # Reduced for demo; increase for real fine-tuning
            per_device_train_batch_size=8,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="no",
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model.save_pretrained('./fine_tuned_model')
        return "Fine-tuning completed successfully!"
    except Exception as e:
        return f"Error during fine-tuning: {str(e)}"

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        
        if "Error" in text:
            st.error(text)
        else:
            st.subheader("Extracted Text")
            st.text_area("Text from PDF", text, height=200)
            
            # Generate embeddings
            sentences, embeddings = generate_embeddings(text)
            
            if isinstance(embeddings, str) and "Error" in embeddings:
                st.error(embeddings)
            else:
                st.subheader("Generated Embeddings")
                st.write(f"Number of sentences: {len(sentences)}")
                st.write(f"Embedding shape: {np.array(embeddings).shape}")
                
                # Display first few embeddings for brevity
                st.write("Sample embeddings (first 3 sentences):")
                for i, (sent, emb) in enumerate(zip(sentences[:3], embeddings[:3])):
                    st.write(f"Sentence {i+1}: {sent}")
                    st.write(f"Embedding (first 5 values): {emb[:5]}")
            
            # Fine-tune model
            st.subheader("Fine-Tuning DistilBERT")
            with st.spinner("Fine-tuning model..."):
                dataset = prepare_dataset(text)
                result = fine_tune_model(dataset)
                st.write(result)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, PyPDF2, SentenceTransformers, and Hugging Face Transformers.")
