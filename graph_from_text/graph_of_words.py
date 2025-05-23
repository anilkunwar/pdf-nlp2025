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
from nltk.tokenize import word_tokenize, sent_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data with error handling."""
    try:
        # Check if punkt_tab and stopwords are already downloaded
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK data already present.")
    except LookupError:
        try:
            logger.info("Downloading NLTK punkt_tab and stopwords...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            logger.info("NLTK data downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {str(e)}")
            st.error(f"Failed to download NLTK data: {str(e)}. Please try again or check your network.")
            return False
    return True

# Download NLTK data at startup
if not download_nltk_data():
    st.stop()

# Set page configuration
st.set_page_config(page_title="PDF Text Extractor & Visualization", layout="wide")

# Title and description
st.title("PDF Text Extractor and Visualization")
st.markdown("""
Upload a PDF file to extract text between specified phrases, generate a word cloud, and create abibliometric network.
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
        logger.error(f"Error extracting text: {str(e)}")
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
        logger.error(f"Error extracting text between phrases: {str(e)}")
        return f"Error extracting text between phrases: {str(e)}"

def generate_word_cloud(text):
    """Generate a word cloud from the provided text."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])  # Add domain-specific stopwords
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        if not filtered_words:
            return None, "No valid words found for word cloud after filtering."
        
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(' '.join(filtered_words))
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(text):
    """Generate a -like keyword co-occurrence network."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])  # Add domain-specific stopwords
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Get word frequencies
        word_freq = Counter(filtered_words)
        if not word_freq:
            return None, "No valid words found for bibliometric network."
        
        # Select top 20 most frequent words as nodes
        top_words = [word for word, freq in word_freq.most_common(20)]
        
        # Create co-occurrence pairs
        sentences = sent_tokenize(text.lower())
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
        return plt.gcf(), None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

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
                wordcloud_fig, wordcloud_error = generate_word_cloud(selected_text)
                if wordcloud_error:
                    st.error(wordcloud_error)
                elif wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                
                # Generate bibliometric network
                st.subheader("Bibliometric Network")
                network_fig, network_error = generate_bibliometric_network(selected_text)
                if network_error:
                    st.error(network_error)
                elif network_fig:
                    st.pyplot(network_fig)

# Footer
st.markdown("---")
st.markdown("Learning the meaning of the words")
