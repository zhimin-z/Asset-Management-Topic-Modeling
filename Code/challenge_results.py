import pandas as pd
import os

path_result = os.path.join(os.path.dirname(os.getcwd()), 'Result')
path_challenge = os.path.join(path_result, 'Challenge')
path_challenge_wordcloud = os.path.join(path_challenge, 'WordCloud')
path_challenge_model = os.path.join(path_challenge, 'Model')

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
umap_model = UMAP(n_components=5, metric='manhattan',
                  random_state=42, low_memory=False)

# Step 3 - Cluster reduced embeddings
min_samples = int(35 * 0.5)
hdbscan_model = HDBSCAN(min_cluster_size=35,
                        min_samples=min_samples, prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

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

df_all = pd.read_json(os.path.join(path_dataset, 'all_filtered.json'))
docs = df_all['Challenge_original_content_gpt_summary'].tolist()

topics, probs = topic_model.fit_transform(docs)
topic_model.save(os.path.join(path_challenge_model, 'Topic model'))

# Preprocess Documents
documents = pd.DataFrame({"Document": docs, "ID": range(len(docs)), "Topic": topics})
documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_challenge_model, 'Topic visualization.html'))

fig = topic_model.visualize_barchart()
fig.write_html(os.path.join(path_challenge_model, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(path_challenge_model, 'Topic similarity visualization.html'))

fig = topic_model.visualize_term_rank()
fig.write_html(os.path.join(path_challenge_model, 'Term score decline visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(cleaned_docs)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_challenge_model, 'Hierarchical clustering visualization.html'))

embeddings = embedding_model.encode(cleaned_docs, show_progress_bar=False)
fig = topic_model.visualize_documents(cleaned_docs, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge_model, 'Document visualization.html'))

info_df = topic_model.get_topic_info()
info_df.to_json(os.path.join(path_challenge_model, 'Topic information.json'),
               indent=4, orient='records')

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics_challenge = topic_model.reduce_outliers(
    docs, topics, probabilities=probs, strategy="probabilities")

df_all['Challenge_topic'] = ''

for index, row in df_all.iterrows():
    df_all.at[index, 'Challenge_topic'] = new_topics_challenge.pop(0)

df_all.to_json(os.path.join(path_dataset, 'all_topics.json'),
               indent=4, orient='records')