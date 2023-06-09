import os
import pickle
import pandas as pd

from bertopic import BERTopic

path_solution_cardsorting = os.path.join(os.getcwd(), 'Result', 'Solution', 'Card Sorting')
path_model = os.path.join(path_solution_cardsorting, 'Model')

path_challenge = os.path.join(path_solution_cardsorting, 'Challenge')
path_extra = os.path.join(path_solution_cardsorting, 'Extra')
path_resolution = os.path.join(path_solution_cardsorting, 'Resolution')

name_model_challenge = 'Challenge_summary_skk6x33z'
name_model_extra = 'Challenge_extra_summary_skk6x33z'
name_model_resolution = 'Resolution_summary_skk6x33z'

df = pd.read_json(os.path.join(path_solution_cardsorting, 'solved.json'))

df['Challenge_summary_topic'] = -1
df['Challenge_extra_summary_topic'] = -1
df['Resolution_summary_topic'] = -1

docs_challenge = []
docs_extra = []
docs_resolution = []

indice_challenge = []
indice_extra = []
indice_resolution = []

column_challenge = '_'.join(name_model_challenge.split('_')[:-1])
column_extra = '_'.join(name_model_extra.split('_')[:-1])
column_resolution = '_'.join(name_model_resolution.split('_')[:-1])

for index, row in df.iterrows():
    if row[column_challenge] != 'N/A':
        indice_challenge.append(index)
        docs_challenge.append(row[column_challenge])
    if row[column_extra] != 'N/A':
        indice_extra.append(index)
        docs_extra.append(row[column_extra])
    if row[column_resolution] != 'N/A':
        indice_resolution.append(index)
        docs_resolution.append(row[column_resolution])
        
topic_model_challenge = BERTopic.load(os.path.join(path_model, name_model_challenge))
topic_model_extra = BERTopic.load(os.path.join(path_model, name_model_extra))
topic_model_resolution = BERTopic.load(os.path.join(path_model, name_model_resolution))

topic_number_challenge = topic_model_challenge.get_topic_info().shape[0] - 1
topic_number_extra = topic_model_extra.get_topic_info().shape[0] - 1
topic_number_resolution = topic_model_resolution.get_topic_info().shape[0] - 1

topics_challenge, probs_challenge = topic_model_challenge.transform(docs_challenge)
topics_extra, probs_extra = topic_model_extra.transform(docs_extra)
topics_resolution, probs_resolution = topic_model_resolution.transform(docs_resolution)

# persist the topic terms
with open(os.path.join(path_challenge, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number_challenge):
        topic_terms.append(topic_model_challenge.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path_extra, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number_extra):
        topic_terms.append(topic_model_extra.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(os.path.join(path_resolution, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number_resolution):
        topic_terms.append(topic_model_resolution.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
fig = topic_model_challenge.visualize_topics()
fig.write_html(os.path.join(path_challenge, 'Topic visualization.html'))

fig = topic_model_extra.visualize_topics()
fig.write_html(os.path.join(path_extra, 'Topic visualization.html'))

fig = topic_model_resolution.visualize_topics()
fig.write_html(os.path.join(path_resolution, 'Topic visualization.html'))

fig = topic_model_challenge.visualize_barchart(top_n_topics=topic_number_challenge, n_words=10)
fig.write_html(os.path.join(path_challenge, 'Term visualization.html'))

fig = topic_model_extra.visualize_barchart(top_n_topics=topic_number_extra, n_words=10)
fig.write_html(os.path.join(path_extra, 'Term visualization.html'))

fig = topic_model_resolution.visualize_barchart(top_n_topics=topic_number_resolution, n_words=10)
fig.write_html(os.path.join(path_resolution, 'Term visualization.html'))

fig = topic_model_challenge.visualize_heatmap()
fig.write_html(os.path.join(path_challenge, 'Topic similarity visualization.html'))

fig = topic_model_extra.visualize_heatmap()
fig.write_html(os.path.join(path_extra, 'Topic similarity visualization.html'))

fig = topic_model_resolution.visualize_heatmap()
fig.write_html(os.path.join(path_resolution, 'Topic similarity visualization.html'))

hierarchical_topics = topic_model_challenge.hierarchical_topics(docs_challenge)
fig = topic_model_challenge.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_challenge, 'Hierarchical clustering visualization.html'))

hierarchical_topics = topic_model_extra.hierarchical_topics(docs_extra)
fig = topic_model_extra.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_extra, 'Hierarchical clustering visualization.html'))

hierarchical_topics = topic_model_resolution.hierarchical_topics(docs_resolution)
fig = topic_model_resolution.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_resolution, 'Hierarchical clustering visualization.html'))

embeddings = topic_model_challenge.embedding_model.embed_documents(docs_challenge)
fig = topic_model_challenge.visualize_documents(docs_challenge, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge, 'Document visualization.html'))

embeddings = topic_model_extra.embedding_model.embed_documents(docs_extra)
fig = topic_model_extra.visualize_documents(docs_extra, embeddings=embeddings)
fig.write_html(os.path.join(path_extra, 'Document visualization.html'))

embeddings = topic_model_resolution.embedding_model.embed_documents(docs_resolution)
fig = topic_model_resolution.visualize_documents(docs_resolution, embeddings=embeddings)
fig.write_html(os.path.join(path_resolution, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
topics_new_challenge = topic_model_challenge.reduce_outliers(docs_challenge, topics_challenge, probabilities=probs_challenge, strategy="probabilities")
topics_new_extra = topic_model_extra.reduce_outliers(docs_extra, topics_extra, probabilities=probs_extra, strategy="probabilities")
topics_new_resolution = topic_model_resolution.reduce_outliers(docs_resolution, topics_resolution, probabilities=probs_resolution, strategy="probabilities")

# persist the document topics
for index, topic in zip(indice_challenge, topics_new_challenge):
    df.at[index, 'Challenge_summary_topic'] = topic

for index, topic in zip(indice_extra, topics_new_extra):
    df.at[index, 'Challenge_extra_summary_topic'] = topic
    
for index, topic in zip(indice_resolution, topics_new_resolution):
    df.at[index, 'Resolution_summary_topic'] = topic
    
# del df['Challenge_summary_topic']
# del df['Challenge_extra_summary_topic']
# del df['Resolution_summary_topic']

df.to_json(os.path.join(path_solution_cardsorting, 'topics.json'), indent=4, orient='records')
