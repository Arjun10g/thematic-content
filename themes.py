import streamlit as st
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from transformers import pipeline
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, pos_tag

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')  # Required for POS tagging

try:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
except Exception as e:
    st.error(f"Error loading Hugging Face models: {e}")

# Load necessary models
sentiment_analyzer = SentimentIntensityAnalyzer()

# Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Default data
default_documents = [
    "Well, thank you very much and thank you, Jeanne. It is my profound honor to be the first President in history to attend the March for Life.",
    "We’re here for a very simple reason: to defend the right of every child, born and unborn, to fulfill their God-given potential.",
    "For 47 years, Americans of all backgrounds have traveled from across the country to stand for life. And today, as President of the United States, I am truly proud to stand with you.",
    "Young people are the heart of the March for Life, and it’s your generation that is making America the pro-family, pro-life nation.",
    "The life movement is led by strong women, amazing faith leaders, and brave students who carry on the legacy of pioneers before us who fought to raise the conscience of our nation and uphold the rights of our citizens.",
    "You embrace mothers with care and compassion. You are powered by prayer and motivated by pure, unselfish love.",
    "All of us here today understand an eternal truth: Every child is a precious and sacred gift from God.",
    "Together, we must protect, cherish, and defend the dignity and sanctity of every human life.",
    "When we see the image of a baby in the womb, we glimpse the majesty of God’s creation.",
    "When we hold a newborn in our arms, we know the endless love that each child brings to a family.",
    "When we watch a child grow, we see the splendor that radiates from each human soul. One life changes the world.",
    "From the first day in office, I’ve taken historic action to support America’s families and to protect the unborn.",
    "Unborn children have never had a stronger defender in the White House.",
    "As the Bible tells us, each person is 'wonderfully made.'",
    "We have taken decisive action to protect religious liberty—so important.",
    "We are protecting pro-life students’ right to free speech on college campuses.",
    "Together, we are the voice for the voiceless.",
    "When it comes to abortion, Democrats have embraced the most radical and extreme positions taken and seen in this country for years.",
    "Nearly every top Democrat in Congress now supports taxpayer-funded abortion, all the way up until the moment of birth.",
    "At the United Nations, I made clear that global bureaucrats have no business attacking the sovereignty of nations that protect innocent life.",
    "We know that every human soul is divine, and every human life—born and unborn—is made in the holy image of Almighty God.",
    "Together, we will defend this truth all across our magnificent land. We will set free the dreams of our people.",
    "I love you all... Thank you. God bless you. And God bless America."
]


# Text input for custom documents
st.sidebar.subheader("Input Your Text Documents")
user_input = st.sidebar.text_area(
    "Enter your own text documents, separated by a new line. Leave as is to use default documents.",
    value="\n".join(default_documents),  # Pre-fill with default documents
    height=200
)

# Process input
if user_input.strip():
    documents = user_input.split("\n")
else:
    documents = default_documents

# Inject custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #2E8B57;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-title {
        font-size: 2em;
        color: #4682B4;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .original-text {
        font-size: 1.2em;
        color: #444;
        background-color: #f9f9f9;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-title">Theme Identification with NLP Methods</div>', unsafe_allow_html=True)

# Help Section
with st.sidebar.expander("Help: Method Explanations", expanded=False):
    st.markdown("""
    **Topic Modeling (LDA):**  
    Latent Dirichlet Allocation (LDA) identifies hidden topics within a collection of documents. It uses word distributions to determine topics and assigns probabilities of each document belonging to these topics.

    **Keyword Extraction (TF-IDF):**  
    TF-IDF (Term Frequency-Inverse Document Frequency) identifies important words by measuring how frequently a term appears in a document compared to how often it appears across all documents.

    **Sentiment Analysis:**  
    Analyzes the sentiment (positive, negative, or neutral) of a text. It leverages lexical resources to assign sentiment scores to words and phrases.

    **Part-of-Speech (POS) Tagging:**  
    Assigns grammatical labels (e.g., noun, verb, adjective) to each word in a text. This helps in understanding the syntactic structure of the text.

    **Text Clustering (KMeans):**  
    Groups similar documents into clusters based on their textual content using KMeans clustering. Useful for unsupervised categorization.

    **Transformer-based (Zero-Shot Classification):**  
    Uses advanced models like BERT or BART to classify text into user-defined categories without requiring any labeled training data.

    **Text Summarization:**  
    Summarizes long documents by extracting or generating concise and informative summaries. Often used for distilling key points.
    """)

# Selector for methodology
method = st.sidebar.selectbox(
    "Choose a Methodology",
    [
        "Topic Modeling (LDA)",
        "Keyword Extraction (TF-IDF)",
        "Sentiment Analysis",
        "Part-of-Speech (POS) Tagging",
        "Text Clustering (KMeans)",
        "Transformer-based (Zero-Shot Classification)",
        "Text Summarization"
    ]
)

# Placeholders for dynamic inputs based on the method
if method == "Topic Modeling (LDA)":
    num_topics = st.sidebar.slider("Number of Topics", 1, 10, 2, key="num_topics")

elif method == "Keyword Extraction (TF-IDF)":
    max_features = st.sidebar.slider("Max Features", 5, 50, 10, key="max_features")

elif method == "Text Clustering (KMeans)":
    num_clusters = st.sidebar.slider("Number of Clusters", 1, 5, 2, key="num_clusters")

elif method == "Transformer-based (Zero-Shot Classification)":
    candidate_labels = st.sidebar.text_area("Enter candidate labels (comma-separated)", "Defense of Life, Religious and Moral Justification, Political Advocacy and Action, Opposition to Political Adversaries, Religious Liberty and Free Speech", key="candidate_labels")

elif method == "Text Summarization":
    max_length = st.sidebar.slider("Summary Max Length", 10, 50, 30, key="max_length")
    min_length = st.sidebar.slider("Summary Min Length", 5, 25, 10, key="min_length")

# Add Run button in the sidebar
run = st.sidebar.button("Run")

if run:  # Execute only when the Run button is clicked
    if method == "Topic Modeling (LDA)":
        st.markdown('<div class="sub-title">Topic Modeling (LDA)</div>', unsafe_allow_html=True)
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(documents)
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(X)

        for idx, topic in enumerate(lda.components_):
            st.write(f"Topic {idx + 1}:")
            st.write([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]])

    elif method == "Keyword Extraction (TF-IDF)":
        st.markdown('<div class="sub-title">Keyword Extraction (TF-IDF)</div>', unsafe_allow_html=True)
        tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
        X = tfidf.fit_transform(documents)
        st.write("Top Keywords:")
        st.write(tfidf.get_feature_names_out())

    elif method == "Sentiment Analysis":
        st.markdown('<div class="sub-title">Sentiment Analysis</div>', unsafe_allow_html=True)
        for doc in documents:
            st.write(doc)
            st.write(sentiment_analyzer.polarity_scores(doc))

    elif method == "Part-of-Speech (POS) Tagging":
        st.markdown('<div class="sub-title">Part-of-Speech (POS) Tagging</div>', unsafe_allow_html=True)
        for doc in documents:
            tokens = word_tokenize(doc)
            pos_tags = pos_tag(tokens)
            st.write(f"Text: {doc}")
            st.write("POS Tags:", pos_tags)

    elif method == "Text Clustering (KMeans)":
        st.markdown('<div class="sub-title">Text Clustering (KMeans)</div>', unsafe_allow_html=True)
        tfidf = TfidfVectorizer(stop_words="english")
        X = tfidf.fit_transform(documents)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        st.write("Cluster Labels:")
        for idx, label in enumerate(kmeans.labels_):
            st.write(f"Document {idx + 1}: Cluster {label}")

    elif method == "Transformer-based (Zero-Shot Classification)":
        st.markdown('<div class="sub-title">Transformer-based Zero-Shot Classification</div>', unsafe_allow_html=True)
        labels = [label.strip() for label in candidate_labels.split(",") if label.strip()]

        if not labels:
            st.warning("Please provide at least one candidate label.")
        else:
            import matplotlib.pyplot as plt
            
            scores_per_label = {label: [] for label in labels}  # Store scores for each label
            
            for idx, doc in enumerate(documents):
                if doc.strip():
                    result = classifier(doc, candidate_labels=labels)
                    scores = result['scores']
                    
                    for label, score in zip(result['labels'], scores):
                        scores_per_label[label].append(score)
                    
                    st.write(f"Text {idx + 1}: {doc[:100]}...")  # Show a short preview of the text

            # Plot scores for each theme
            st.markdown('<div class="sub-title">Theme Classification Scores (Line Plot)</div>', unsafe_allow_html=True)
            
            plt.figure(figsize=(10, 6))
            for label, scores in scores_per_label.items():
                plt.plot(range(1, len(scores) + 1), scores, marker='o', label=label)
            
            plt.xlabel("Document Index")
            plt.ylabel("Classification Score")
            plt.title("Zero-Shot Classification Scores by Theme")
            plt.legend(title="Themes")
            plt.grid(True)
            
            # Show the plot
            st.pyplot(plt)

    elif method == "Text Summarization":
        st.markdown('<div class="sub-title">Text Summarization</div>', unsafe_allow_html=True)
        for doc in documents:
            summary = summarizer(doc, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            st.write(f"Text: {doc}")
            st.write("Summary:", summary)


# pwd
# cd "/Users/arjunghumman/Downloads/VS Code Stuff/Python/Thematic Identifier"
# streamlit run themes.py