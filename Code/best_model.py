
from bertopic import BERTopic
from matplotlib import pyplot as plt
from wordcloud import WordCloud

import os
import pickle
import pandas as pd

path_result = 'Result'
path_dataset = 'Dataset'

path_general = os.path.join(path_result, 'General')
path_challenge = os.path.join(path_result, 'Challenge')
path_solution = os.path.join(path_result, 'Solution')

path_model_challenge = os.path.join(path_challenge, 'Model')
path_model_solution = os.path.join(path_solution, 'Model')

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

# output the best topic model on challenges

model_challenge = 'Challenge_gpt_summary_6dss4sq4'
column_challenge = '_'.join(model_challenge.split('_')[:-1])

df['Challenge_topic'] = -1

indice_challenge = []
docs_challenge = []

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]):
        indice_challenge.append(index)
        docs_challenge.append(row[column_challenge])

topic_model = BERTopic.load(os.path.join(
    path_model_challenge, model_challenge))
topics, probs = topic_model.transform(docs_challenge)

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(
    path_challenge, 'Topic information.json'), indent=4, orient='records')

# persist the topic terms
topic_terms = []
for i in range(df_topics.shape[0] - 1):
    topic_terms.append(topic_model.get_topic(i))

with open(os.path.join(path_challenge, 'Topic terms.pickle'), 'wb') as handle:
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

embeddings = topic_model.embedding_model.embed_documents(docs_challenge)
fig = topic_model.visualize_documents(docs_challenge, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics_challenge = topic_model.reduce_outliers(
    docs_challenge, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_challenge, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {'Document': docs_challenge, 'Topic': new_topics_challenge})
documents_per_topic = documents.groupby(['Topic']).agg(
    {'Document': ' '.join}).reset_index()

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

model_solution = 'Solution_gpt_summary_zvuf1veb'
column_solution = '_'.join(model_solution.split('_')[:-1])

df['Solution_topic'] = -1

indice_solution = []
docs_solution = []

for index, row in df.iterrows():
    if pd.notna(row[column_solution]):
        indice_solution.append(index)
        docs_solution.append(row[column_solution])

topic_model = BERTopic.load(os.path.join(path_model_solution, model_solution))
topics, probs = topic_model.transform(docs_solution)

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(
    path_solution, 'Topic information.json'), indent=4, orient='records')

# persist the topic terms
topic_terms = []
for i in range(df_topics.shape[0] - 1):
    topic_terms.append(topic_model.get_topic(i))

with open(os.path.join(path_solution, 'Topic terms.pickle'), 'wb') as handle:
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

embeddings = topic_model.embedding_model.embed_documents(docs_solution)
fig = topic_model.visualize_documents(docs_solution, embeddings=embeddings)
fig.write_html(os.path.join(path_solution, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
new_topics_solution = topic_model.reduce_outliers(
    docs_solution, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_solution, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {'Document': docs_solution, 'Topic': new_topics_solution})
documents_per_topic = documents.groupby('Topic').agg(
    {'Document': ' '.join}).reset_index()

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
