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
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import spacy
from math import log

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK and spaCy data
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

# Download spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy en_core_web_sm model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Download NLTK data at startup
if not download_nltk_data():
    st.stop()

# Simplified IDF approximation for scientific texts
IDF_APPROX = {
    # Common scientific terms with lower IDF (less unique)
    "study": log(1000 / 800), "analysis": log(1000 / 700), "results": log(1000 / 600),
    "method": log(1000 / 500), "experiment": log(1000 / 400),
    # Specific terms with higher IDF (more unique)
    "spectroscopy": log(1000 / 50), "nanoparticle": log(1000 / 40), "diffraction": log(1000 / 30),
    "microscopy": log(1000 / 20), "quantum": log(1000 / 10)
    # Default IDF for unknown terms: log(N / 100), assuming 1000 documents, 100 occurrences
}
DEFAULT_IDF = log(1000 / 100)

# Define keyword categories for physics/materials science
KEYWORD_CATEGORIES = {
    "Materials": [
        "alloy", "polymer", "nanoparticle", "crystal", "metal", "ceramic", "composite", "semiconductor",
        "graphene", "nanotube", "oxide", "thin film", "superconductor", "biomaterial"
    ],
    "Methods/Techniques": [
        "spectroscopy", "diffraction", "microscopy", "lithography", "deposition", "etching", "annealing",
        "characterization", "synthesis", "fabrication", "imaging", "scanning", "tomography"
    ],
    "Physical Phenomena": [
        "diffusion", "scattering", "conductivity", "magnetism", "superconductivity", "fluorescence",
        "polarization", "refraction", "absorption", "emission", "quantum", "thermal"
    ],
    "Properties": [
        "hardness", "conductivity", "resistivity", "magnetization", "density", "strength", "elasticity",
        "viscosity", "porosity", "permeability", "ductility", "toughness"
    ],
    "Other": []
}

# Set page configuration
st.set_page_config(page_title="PDF Text Extractor & Visualization", layout="wide")

# Title and description
st.title("PDF Text Extractor and Visualization")
st.markdown("""
Upload a PDF file to extract text between specified phrases, configure keyword selection criteria, and generate publication-quality word clouds and colorful bibliometric networks.
The app extracts text using PyPDF2, processes keywords with NLTK and spaCy, and visualizes using WordCloud and NetworkX.
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

def get_candidate_keywords(text, min_freq, min_length, use_stopwords, custom_stopwords, top_limit, tfidf_weight, use_nouns_only, include_phrases):
    """Extract and categorize candidate keywords with user-specified criteria."""
    stop_words = set(stopwords.words('english')) if use_stopwords else set()
    stop_words.update(['laser', 'microstructure', 'introduction', 'conclusion', 'section', 'chapter', 'figure', 'table'])
    stop_words.update([w.lower() for w in custom_stopwords.split(",") if w.strip()])

    # Extract single words
    words = word_tokenize(text.lower())
    if use_nouns_only:
        doc = nlp(text)
        nouns = {token.text.lower() for token in doc if token.pos_ == "NOUN"}
        filtered_words = [w for w in words if w in nouns and w.isalnum() and len(w) >= min_length and w not in stop_words]
    else:
        filtered_words = [w for w in words if w.isalnum() and len(w) >= min_length and w not in stop_words]
    
    word_freq = Counter(filtered_words)
    
    # Extract phrases if enabled
    phrases = []
    if include_phrases:
        doc = nlp(text)
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and len(chunk.text) >= min_length]
        phrase_freq = Counter(phrases)
        # Filter phrases by frequency
        phrases = [(p, f) for p, f in phrase_freq.items() if f >= min_freq]

    # Compute TF-IDF scores
    total_words = len(word_tokenize(text))
    tfidf_scores = {}
    for word, freq in word_freq.items():
        if freq < min_freq:
            continue
        tf = freq / total_words
        idf = IDF_APPROX.get(word, DEFAULT_IDF)
        tfidf_scores[word] = tf * idf * tfidf_weight
    
    for phrase, freq in phrases:
        if freq < min_freq:
            continue
        tf = freq / total_words
        # Approximate IDF for phrases by using the IDF of the head noun
        head_noun = phrase.split()[-1]
        idf = IDF_APPROX.get(head_noun, DEFAULT_IDF)
        tfidf_scores[phrase] = tf * idf * tfidf_weight

    # Rank by TF-IDF if weight > 0, else by frequency
    if tfidf_weight > 0:
        ranked_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_limit]
    else:
        ranked_terms = [(w, f) for w, f in word_freq.most_common(top_limit) if f >= min_freq]
        ranked_terms += phrases[:top_limit - len(ranked_terms)]
    
    # Categorize keywords
    categorized_keywords = {cat: [] for cat in KEYWORD_CATEGORIES}
    for term, _ in ranked_terms:
        for category, keywords in KEYWORD_CATEGORIES.items():
            if term in keywords or any(k in term for k in keywords):  # Match phrases containing category keywords
                categorized_keywords[category].append(term)
                break
        else:
            categorized_keywords["Other"].append(term)
    
    return categorized_keywords

def generate_word_cloud(text, selected_keywords, selection_criteria):
    """Generate a publication-quality word cloud with caption."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word in selected_keywords]
        
        if not filtered_words:
            return None, "No valid words found for word cloud after filtering."
        
        wordcloud = WordCloud(
            width=1200, height=600,
            background_color='white',
            min_font_size=8,
            max_font_size=150,
            font_path=None,
            colormap='viridis'
        ).generate(' '.join(filtered_words))
        
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Selected Keywords", fontsize=14, pad=10)
        # Add caption
        caption = f"Word Cloud generated with: {selection_criteria}"
        plt.figtext(0.5, 0.01, caption, ha="center", fontsize=10, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig, None
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        return None, f"Error generating word cloud: {str(e)}"

def generate_bibliometric_network(text, selected_keywords, label_font_size, selection_criteria):
    """Generate a publication-quality colorful keyword co-occurrence network with enhanced aesthetics."""
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['laser', 'microstructure'])
        words = word_tokenize(text.lower())
        filtered_words = [word for word in words if word in selected_keywords]
        
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
        
        # Color nodes by community
        communities = greedy_modularity_communities(G)
        node_colors = {}
        palette = sns.color_palette("viridis", len(communities))
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = palette[i]
        
        # Scale edge weights for thickness (logarithmic for better differentiation)
        edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
        max_weight = max(edge_weights, default=1)
        edge_widths = [2.5 * np.log1p(weight) + 1 for weight in edge_weights]  # Logarithmic scaling
        edge_colors = [plt.cm.Blues(weight / max_weight) for weight in edge_weights]
        
        # Set up plot with enhanced aesthetics
        plt.style.use('seaborn-v0_8-white')
        plt.rcParams['font.family'] = 'Arial'
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        pos = nx.spring_layout(G, k=0.6, seed=42)
        
        # Node sizes scaled more prominently
        node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes]
        
        # Draw nodes with borders
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[node_colors[node] for node in G.nodes],
            edgecolors='black',  # Add black borders
            linewidths=1.0,  # Border thickness
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with enhanced thickness
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            ax=ax
        )
        
        # Draw labels with background halo
        nx.draw_networkx_labels(
            G, pos,
            font_size=label_font_size,
            font_weight='bold',
            font_color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'),
            ax=ax
        )
        
        ax.set_title("Keyword Co-occurrence Network", fontsize=16, pad=15, fontweight='bold')
        
        # Add caption with professional styling
        caption = f"Keyword co-occurrence network generated with: {selection_criteria}"
        plt.figtext(
            0.5, 0.02,
            caption,
            ha="center",
            fontsize=10,
            wrap=True,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        ax.set_facecolor('#f5f5f5')  # Light gray background for contrast
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
                
                # Keyword selection configuration
                st.subheader("Configure Keyword Selection Criteria")
                min_freq = st.slider("Minimum frequency", min_value=1, max_value=10, value=2, help="Minimum occurrences of a word/phrase")
                min_length = st.slider("Minimum length", min_value=3, max_value=10, value=4, help="Minimum characters in a word/phrase")
                use_stopwords = st.checkbox("Use stopword filtering", value=True, help="Remove common English words (e.g., 'the', 'is')")
                custom_stopwords = st.text_input("Custom stopwords (comma-separated)", "", help="Add specific words to exclude (e.g., 'study,results')")
                top_limit = st.slider("Top limit (max keywords)", min_value=10, max_value=100, value=50, step=10, help="Maximum number of candidate keywords")
                tfidf_weight = st.slider("TF-IDF weighting (statistical relevance)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, help="Higher values prioritize rare, significant terms")
                use_nouns_only = st.checkbox("Filter for nouns only (linguistic filtering)", value=False, help="Include only nouns for more specific terms")
                include_phrases = st.checkbox("Include multi-word phrases", value=True, help="Extract noun phrases like 'laser ablation'")
                
                # Generate selection criteria caption
                criteria_parts = []
                criteria_parts.append(f"frequency ≥ {min_freq}")
                criteria_parts.append(f"length ≥ {min_length}")
                criteria_parts.append("stopwords " + ("enabled" if use_stopwords else "disabled"))
                if custom_stopwords.strip():
                    criteria_parts.append(f"custom stopwords: {custom_stopwords}")
                criteria_parts.append(f"top {top_limit} keywords")
                criteria_parts.append(f"TF-IDF weight: {tfidf_weight}")
                criteria_parts.append("nouns only" if use_nouns_only else "all parts of speech")
                criteria_parts.append("multi-word phrases " + ("included" if include_phrases else "excluded"))
                selection_criteria = ", ".join(criteria_parts)
                
                # Keyword selection by category
                st.subheader("Select Keywords by Category")
                categorized_keywords = get_candidate_keywords(
                    selected_text, min_freq, min_length, use_stopwords, custom_stopwords, top_limit, tfidf_weight, use_nouns_only, include_phrases
                )
                selected_keywords = []
                
                for category, keywords in categorized_keywords.items():
                    if keywords:
                        with st.expander(f"{category} ({len(keywords)} keywords)"):
                            selected = st.multiselect(
                                f"Select keywords from {category}",
                                options=keywords,
                                default=keywords[:min(5, len(keywords))]
                            )
                            selected_keywords.extend(selected)
                
                if not selected_keywords:
                    st.error("Please select at least one keyword.")
                    st.stop()
                
                # Font size selection for network labels
                st.subheader("Network Visualization Settings")
                label_font_size = st.slider("Select font size for network labels", min_value=8, max_value=20, value=10, step=1)
                
                # Generate word cloud
                st.subheader("Word Cloud")
                wordcloud_fig, wordcloud_error = generate_word_cloud(selected_text, selected_keywords, selection_criteria)
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
                network_fig, network_error = generate_bibliometric_network(selected_text, selected_keywords, label_font_size, selection_criteria)
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
st.markdown("Built with Streamlit, PyPDF2, WordCloud, NetworkX, NLTK, spaCy, Matplotlib, and Seaborn.")