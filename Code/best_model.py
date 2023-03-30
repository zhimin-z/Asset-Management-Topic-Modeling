from wordcloud import WordCloud
from matplotlib import pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import pandas as pd

path_result = 'Result'
path_dataset = 'Dataset'

name_challenge = 'Challenge_gpt_summary'
name_solution = 'Solution_gpt_summary'

path_general = os.path.join(path_result, 'General')
path_challenge = os.path.join(path_result, 'Challenge')
path_solution = os.path.join(path_result, 'Solution')

# set default sweep configuration
config_defaults = {
    # Refer to https://www.sbert.net/docs/pretrained_models.html
    'model_name': 'all-MiniLM-L6-v2',
    'metric_distane': 'manhattan',
    'n_components': 5,
    'min_samples': 5,
    'low_memory': False,
    'reduce_frequent_words': True,
}

config_challenge = {
    'min_cluster_size': 30,
    'min_samples_pct': 0.1,
    'ngram_range': 2,
}

config_solution = {
    'min_cluster_size': 20,
    'min_samples_pct': 0.3,
    'ngram_range': 2,
}

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

# run best challenge topic model

df['Challenge_topic'] = -1

indexes_challenge = []
docs_challenge = []

for index, row in df.iterrows():
    if pd.notna(row[name_challenge]) and len(row[name_challenge].split()) >= 5:
        indexes_challenge.append(index)
        docs_challenge.append(row[name_challenge])

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer(config_defaults['model_name'])

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_components=config_defaults['n_components'],
                  metric=config_defaults['metric_distane'], low_memory=config_defaults['low_memory'])

# Step 3 - Cluster reduced embeddings
samples = int(config_challenge['min_cluster_size']
                  * config_challenge['min_samples_pct'])
min_samples = samples if samples > config_defaults['min_samples'] else config_defaults['min_samples']
hdbscan_model = HDBSCAN(
    min_cluster_size=config_challenge['min_cluster_size'], min_samples=min_samples, prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = TfidfVectorizer(
    ngram_range=(1, config_challenge['ngram_range']))

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=config_defaults['reduce_frequent_words'])

# Step 6 - (Optional) Fine-tune topic representation
representation_model = KeyBERTInspired()

# All steps together
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(docs_challenge)

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(
    path_challenge, 'Topic information.json'), indent=4, orient='records')

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_challenge, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(
    top_n_topics=df_topics.shape[0]-1, n_words=10)
fig.write_html(os.path.join(path_challenge, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(
    path_challenge, 'Topic similarity visualization.html'))

fig = topic_model.visualize_term_rank()
fig.write_html(os.path.join(
    path_challenge, 'Term score decline visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(docs_challenge)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(
    path_challenge, 'Hierarchical clustering visualization.html'))

embeddings = embedding_model.encode(docs_challenge, show_progress_bar=False)
fig = topic_model.visualize_documents(docs_challenge, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics = topic_model.reduce_outliers(
    docs_challenge, topics, probabilities=probs, strategy="probabilities")

for index in indexes_challenge:
    df.at[index, 'Challenge_topic'] = new_topics.pop(0)

path_wordcloud = os.path.join(path_challenge, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {"Document": docs_challenge, "ID": range(len(docs_challenge)), "Topic": topics})
documents_per_topic = documents.groupby(
    ['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(
    documents_per_topic.Document.values)

for index, doc in enumerate(cleaned_docs):
    wordcloud = WordCloud(
        width=800, height=800, background_color='white', min_font_size=10).generate(doc)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud,
                f'Topic_{index}'+'.png'), bbox_inches='tight')
    plt.close()

# run best solution topic model

df['Solution_topic'] = -1

indexes_solution = []
docs_solution = []

for index, row in df.iterrows():
    if pd.notna(row[name_solution]) and len(row[name_solution].split()) >= 5:
        indexes_solution.append(index)
        docs_solution.append(row[name_solution])

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer(config_defaults['model_name'])

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_components=config_defaults['n_components'],
                  metric=config_defaults['metric_distane'], low_memory=config_defaults['low_memory'])

# Step 3 - Cluster reduced embeddings
min_samples = int(config_solution['min_cluster_size']
                  * config_solution['min_samples_pct'])
min_samples = 5 if min_samples < 5 else min_samples
hdbscan_model = HDBSCAN(
    min_cluster_size=config_solution['min_cluster_size'], min_samples=min_samples, prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = TfidfVectorizer(
    ngram_range=(1, config_solution['ngram_range']))

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer(
    reduce_frequent_words=config_defaults['reduce_frequent_words'])

# Step 6 - Fine-tune topic representation
representation_model = KeyBERTInspired()

# All steps together
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    calculate_probabilities=True
)

topics, probs = topic_model.fit_transform(docs_solution)

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(
    path_solution, 'Topic information.json'), indent=4, orient='records')

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_solution, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(
    top_n_topics=df_topics.shape[0]-1, n_words=10)
fig.write_html(os.path.join(path_solution, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(
    path_solution, 'Topic similarity visualization.html'))

fig = topic_model.visualize_term_rank()
fig.write_html(os.path.join(
    path_solution, 'Term score decline visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(docs_solution)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(
    path_solution, 'Hierarchical clustering visualization.html'))

embeddings = embedding_model.encode(docs_solution, show_progress_bar=False)
fig = topic_model.visualize_documents(docs_solution, embeddings=embeddings)
fig.write_html(os.path.join(path_solution, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics = topic_model.reduce_outliers(
    docs_solution, topics, probabilities=probs, strategy="probabilities")

for index in indexes_solution:
    df.at[index, 'Solution_topic'] = new_topics.pop(0)

del df['Challenge_original_content']
del df['Challenge_preprocessed_content']
del df['Challenge_gpt_summary']

del df['Solution_original_content']
del df['Solution_preprocessed_content']
del df['Solution_gpt_summary']

df.to_json(os.path.join(path_general, 'Topics.json'),
           indent=4, orient='records')

path_wordcloud = os.path.join(path_solution, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {"Document": docs_solution, "ID": range(len(docs_solution)), "Topic": topics})
documents_per_topic = documents.groupby(
    ['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(
    documents_per_topic.Document.values)

for index, doc in enumerate(cleaned_docs):
    wordcloud = WordCloud(
        width=800, height=800, background_color='white', min_font_size=10).generate(doc)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud,
                f'Topic_{index}'+'.png'), bbox_inches='tight')
    plt.close()