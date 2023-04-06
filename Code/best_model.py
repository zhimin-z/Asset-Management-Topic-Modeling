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

path_general = os.path.join(path_result, 'General')
path_challenge = os.path.join(path_result, 'Challenge')
path_solution = os.path.join(path_result, 'Solution')

path_model_challenge = os.path.join(path_challenge, 'Model')
path_model_solution = os.path.join(path_solution, 'Model')

model_challenge = 'Challenge_gpt_summary_5y1kc0q0'
model_solution = 'Solution_gpt_summary_wmng21rs'

column_challenge = ' '.join(model_challenge.split('_')[:-1])
column_solution = ' '.join(model_solution.split('_')[:-1])

# set default sweep configuration
config_defaults = {
    # Refer to https://www.sbert.net/docs/pretrained_models.html
    'model_name': 'all-MiniLM-L6-v2',
    'metric_distane': 'manhattan',
    'calculate_probabilities': True,
    'reduce_frequent_words': True,
    'prediction_data': True,
    'low_memory': False,
    'random_state': 42,
    'n_components': 5,
    'min_samples': 5,
}

config_challenge = {
    'min_cluster_size': 30,
    'min_samples_pct': 0.3,
    'ngram_range': 2,
}

config_solution = {
    'min_cluster_size': 20,
    'min_samples_pct': 0.2,
    'ngram_range': 2,
}

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

# output the best topic model on challenges

df['Challenge_topic'] = -1

indice_challenge = []
docs_challenge = []

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]):
        indice_challenge.append(index)
        docs_challenge.append(row[column_challenge])

topic_model = BERTopic.load(model_challenge)
topics, probs = topic_model.transform(docs_challenge)

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
new_topics_challenge = topic_model.reduce_outliers(
    docs_challenge, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_challenge, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame({'Document': docs_challenge, 'Topic': new_topics_challenge})
documents_per_topic = documents.groupby(['Topic']).agg({'Document': ' '.join}).reset_index()

for index, row in documents_per_topic.iterrows():
    wordcloud = WordCloud(
        width=800, height=800, background_color='white', min_font_size=10).generate(row['Document'])
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud,
                f'Topic_{row["Topic"]}'+'.png'), bbox_inches='tight')
    plt.close()

# output the best topic model on solutions

df['Solution_topic'] = -1

indice_solution = []
docs_solution = []

for index, row in df.iterrows():
    if pd.notna(row[column_solution]):
        indice_solution.append(index)
        docs_solution.append(row[column_solution])

topic_model = BERTopic.load(model_solution)
topics, probs = topic_model.transform(docs_solution)

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
new_topics_solution = topic_model.reduce_outliers(
    docs_solution, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_solution, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame({'Document': docs_solution, 'Topic': new_topics_solution})
documents_per_topic = documents.groupby('Topic').agg({'Document': ' '.join}).reset_index()

for index, row in documents_per_topic.iterrows():
    wordcloud = WordCloud(
        width=800, height=800, background_color='white', min_font_size=10).generate(row['Document'])
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud,
                f'Topic_{row["Topic"]}'+'.png'), bbox_inches='tight')
    plt.close()
    
# persist the document topics

for index, topic in zip(indice_challenge, new_topics_challenge):
    df.at[index, 'Challenge_topic'] = topic

for index, topic in zip(indice_solution, new_topics_solution):
    df.at[index, 'Solution_topic'] = topic

del df['Challenge_original_content']
del df['Challenge_preprocessed_content']
del df['Challenge_gpt_summary']

del df['Solution_original_content']
del df['Solution_preprocessed_content']
del df['Solution_gpt_summary']

df.to_json(os.path.join(path_general, 'Topics.json'),
           indent=4, orient='records')