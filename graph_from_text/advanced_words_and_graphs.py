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
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data with error handling."""
    try:
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
Upload a PDF file to extract text between specified phrases, select relevant keywords, and generate publication-quality word clouds and bibliometric networks.
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

def get_candidate_keywords(text):
    """Extract candidate keywords with scientific relevance."""
    stop_words = set(stopwords.words('english'))
    stop_words.update(['laser', 'microstructure', 'introduction', 'conclusion', 'section', 'chapter', 'figure', 'table'])
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    # Select top 50 words with frequency >= 2 for candidate keywords
    return [word for word, freq in word_freq.most_common(50) if freq >= 2]

def generate_word_cloud(text, selected_keywords):
    """Generate a publication-quality word cloud from the provided text."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and word in selected_keywords]
        
        if not filtered_words:
            return None, "No valid words found for word cloud after filtering."
        
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            min_font_size=8,
            max_font_size=150,
            font_path=None,  # Use default font (publication-safe)
            colormap='viridis'  # Professional colormap
        ).generate(' '.join(filtered_words))
        
        plt.style.use('seaborn-v0_8')  # Updated to valid Matplotlib style
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # High resolution
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Selected Keywords", fontsize=14, pad=10)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(text, selected_keywords):
    """Generate a publication-quality keyword co-occurrence network."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and word in selected_keywords]
        
        word_freq = Counter(filtered_words)
        if not word_freq:
            return None, "No valid words found for bibliometric network."
        
        top_words = [word for word, freq in word_freq.most_common(20)]
        
        sentences = sent_tokenize(text.lower())
        co_occurrences = Counter()
        for sentence in sentences:
            words_in_sentence = [word for word in word_tokenize(sentence) if word in top_words]
            for pair in combinations(set(words_in_sentence), 2):
                co_occurrences[tuple(sorted(pair))] += 1
        
        G = nx.Graph()
        for word, freq in word_freq.most_common(20):
            G.add_node(word, size=freq)
        
        for (word1, word2), weight in co_occurrences.items():
            if word1 in top_words and word2 in top_words:
                G.add_edge(word1, word2, weight=weight)
        
        plt.style.use('seaborn-v0_8')  # Updated to valid Matplotlib style
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)  # High resolution
        pos = nx.spring_layout(G, k=0.5, seed=42)  # Consistent layout
        node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes]
        edge_weights = [G.edges[edge]['weight'] * 0.5 for edge in G.edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        ax.set_title("Keyword Co-occurrence Network", fontsize=14, pad=10)
        plt.tight_layout()
        return fig, None
    except Exception as e:
        logger.error(f"Error generating bibliometric network: {str(e)}")
        return None, f"Error generating bibliometric network: {str(e)}"

def save_figure(fig, filename):
    """Save figure as PNG and SVG for publication."""
    try:
        fig.savefig(filename + ".png", dpi=300, bbox_inches='tight', format='png')
        fig.savefig(filename + ".svg", bbox_inches='tight', format='svg')
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        return False

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
                
                # Keyword selection
                st.subheader("Select Keywords")
                candidate_keywords = get_candidate_keywords(selected_text)
                if not candidate_keywords:
                    st.warning("No candidate keywords found. Using all words.")
                    selected_keywords = set(word_tokenize(selected_text.lower()))
                else:
                    selected_keywords = st.multiselect(
                        "Select keywords to include (deselect to exclude):",
                        options=candidate_keywords,
                        default=candidate_keywords[:10]  # Default to top 10
                    )
                    if not selected_keywords:
                        st.error("Please select at least one keyword.")
                        st.stop()
                
                # Generate word cloud
                st.subheader("Word Cloud")
                wordcloud_fig, wordcloud_error = generate_word_cloud(selected_text, selected_keywords)
                if wordcloud_error:
                    st.error(wordcloud_error)
                elif wordcloud_fig:
                    st.pyplot(wordcloud_fig)
                    if save_figure(wordcloud_fig, "wordcloud"):
                        st.download_button(
                            label="Download Word Cloud (PNG)",
                            data=open("wordcloud.png", "rb").read(),
                            file_name="wordcloud.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Word Cloud (SVG)",
                            data=open("wordcloud.svg", "rb").read(),
                            file_name="wordcloud.svg",
                            mime="image/svg+xml"
                        )
                
                # Generate bibliometric network
                st.subheader("Bibliometric Network")
                network_fig, network_error = generate_bibliometric_network(selected_text, selected_keywords)
                if network_error:
                    st.error(network_error)
                elif network_fig:
                    st.pyplot(network_fig)
                    if save_figure(network_fig, "network"):
                        st.download_button(
                            label="Download Network (PNG)",
                            data=open("network.png", "rb").read(),
                            file_name="network.png",
                            mime="image/png"
                        )
                        st.download_button(
                            label="Download Network (SVG)",
                            data=open("network.svg", "rb").read(),
                            file_name="network.svg",
                            mime="image/svg+xml"
                        )

# Footer
st.markdown("---")
st.markdown("Intuitively learning the meaning of words.")
