import os
import pandas as pd

path_dataset = 'Dataset'
df_all = pd.read_json(os.path.join(path_dataset, 'all_topics.json'))
df_all = df_all[df_all['Solution_original_content'].isnull() == False]
df_all = df_all[df_all['Solution_original_content'] != '']
docs = df_all['Solution_original_content_gpt_summary'].tolist()

# visualize the best challenge topic model

from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_components=5, metric='manhattan', low_memory=False)

# Step 3 - Cluster reduced embeddings
min_samples = int(15 * 0.25)
hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=min_samples, prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = TfidfVectorizer(ngram_range=(1, 2))

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

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

topics, probs = topic_model.fit_transform(docs)

path_result = 'Result'
path_solution = os.path.join(path_result, 'Solution')

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(path_solution, 'Topic information.json'), indent=4, orient='records')

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_solution, 'Topic visualization.html'))

fig = topic_model.visualize_barchart()
fig.write_html(os.path.join(path_solution, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(path_solution, 'Topic similarity visualization.html'))

fig = topic_model.visualize_term_rank()
fig.write_html(os.path.join(path_solution, 'Term score decline visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(docs)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_solution, 'Hierarchical clustering visualization.html'))

embeddings = embedding_model.encode(docs, show_progress_bar=False)
fig = topic_model.visualize_documents(docs, embeddings=embeddings)
fig.write_html(os.path.join(path_solution, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics = topic_model.reduce_outliers(
    docs, topics, probabilities=probs, strategy="probabilities")

df_all['Solution_topic'] = -1

for index, row in df_all.iterrows():
    if not row['Solution_original_content_gpt_summary']:
        continue
    df_all.at[index, 'Solution_topic'] = new_topics.pop(0)

df_all.to_json(os.path.join(path_dataset, 'all_topics.json'), indent=4, orient='records')

from matplotlib import pyplot as plt
from wordcloud import WordCloud

path_wordcloud = os.path.join(path_solution, 'Wordcloud')

# Preprocess Documents
documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

for index, doc in enumerate(cleaned_docs):
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(doc)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud, f'Topic_{index}'+'.png'), bbox_inches='tight')
    plt.close()