import streamlit as st
import PyPDF2
import tempfile
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(page_title="PDF Text Extractor & Visualization", layout="wide")

# Title and description
st.title("PDF Text Extractor and Visualization")
st.markdown("""
Upload a PDF file to extract text between specified phrases, generate a word cloud and bibliometric network.
The app extracts text using PyPDF2, creates a word cloud with WordCloud, and generates a keyword co-occurrence network using NetworkX.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Input fields for desired phrases
start_phrase = st.text_input("Enter the desired initial phrase", "Introduction")
end_phrase = st.text_input("Enter the desired final phrase", "Conclusion")

def extract_text_from_pdf(file):
    """Extract text from a PDF file using PyPDF2."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        pdf_reader = PyPDF2.PdfReader(tmp_file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        os.unlink(tmp_file_path)
        return text if text.strip() else "No text extracted from the PDF."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_between_phrases(text, start_phrase, end_phrase):
    """Extract text between start_phrase and end_phrase."""
    try:
        start_idx = text.find(start_phrase)
        end_idx = text.find(end_phrase, start_idx + len(start_phrase))
        if start_idx == -1 or end_idx == -1:
            return "Specified phrases not found in the text."
        return text[start_idx:end_idx + len(end_phrase)]
    except Exception as e:
        return f"Error extracting text between phrases: {str(e)}"

def generate_word_cloud(text):
    """Generate a word cloud from the provided text."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(['laser', 'microstructure'])  # Add domain-specific stopwords if needed
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(' '.join(filtered_words))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def generate_bibliometric_network(text):
    """Generate a VOSviewer-like keyword co-occurrence network."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(['laser', 'microstructure'])  # Add domain-specific stopwords
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Get word frequencies
    word_freq = Counter(filtered_words)
    # Select top 20 most frequent words as nodes
    top_words = [word for word, freq in word_freq.most_common(20)]
    
    # Create co-occurrence pairs
    sentences = nltk.sent_tokenize(text.lower())
    co_occurrences = Counter()
    for sentence in sentences:
        words_in_sentence = [word for word in word_tokenize(sentence) if word in top_words]
        for pair in combinations(set(words_in_sentence), 2):
            co_occurrences[tuple(sorted(pair))] += 1
    
    # Create network
    G = nx.Graph()
    for word, freq in word_freq.most_common(20):
        G.add_node(word, size=freq)
    
    for (word1, word2), weight in co_occurrences.items():
        if word1 in top_words and word2 in top_words:
            G.add_edge(word1, word2, weight=weight)
    
    # Draw network
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5)
    node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes]
    edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in edge_weights], alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Keyword Co-occurrence Network")
    return plt.gcf()

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Extract text
        text = extract_text_from_pdf(uploaded_file)
        
        if "Error" in text:
            st.error(text)
        else:
            # Extract text between phrases
            selected_text = extract_text_between_phrases(text, start_phrase, end_phrase)
            
            if "Error" in selected_text or "not found" in selected_text:
                st.error(selected_text)
            else:
                st.subheader("Extracted Text Between Phrases")
                st.text_area("Selected Text", selected_text, height=200)
                
                # Generate word cloud
                st.subheader("Word Cloud")
                wordcloud_fig = generate_word_cloud(selected_text)
                st.pyplot(wordcloud_fig)
                
                # Generate bibliometric network
                st.subheader("Bibliometric Network (VOSviewer-like)")
                network_fig = generate_bibliometric_network(selected_text)
                st.pyplot(network_fig)

# Footer
st.markdown("---")
st.markdown("Learning the meaning of words.")
