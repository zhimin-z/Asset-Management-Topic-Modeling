import os
import pickle
import pandas as pd

from bertopic import BERTopic

path_output = os.path.join('Result', 'RQ1', 'General Topics')
path_topic = os.path.join('Code', 'RQ1', 'General Topic Modeling')
path_model = os.path.join(path_topic, 'Model')

name_model_challenge = 'Challenge_gpt_summary_preprocessed_content_7k6s9mi8'
column_challenge = '_'.join(name_model_challenge.split('_')[:-1])

df = pd.read_json(os.path.join('Dataset', 'preprocessed.json'))
df['Challenge_topic'] = -1

indice = []
docs = []

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]) and len(row[column_challenge]):
        indice.append(index)
        docs.append(row[column_challenge])
        
topic_model_challenge = BERTopic.load(os.path.join(path_model, name_model_challenge))
topic_number_challenge = topic_model_challenge.get_topic_info().shape[0] - 1
topics_challenge, probs_challenge = topic_model_challenge.transform(docs)

# persist the topic terms
with open(os.path.join(path_topic, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number_challenge):
        topic_terms.append(topic_model_challenge.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model_challenge.visualize_topics()
fig.write_html(os.path.join(path_topic, 'Topic visualization.html'))

fig = topic_model_challenge.visualize_barchart(top_n_topics=topic_number_challenge, n_words=10)
fig.write_html(os.path.join(path_topic, 'Term visualization.html'))

fig = topic_model_challenge.visualize_heatmap()
fig.write_html(os.path.join(path_topic, 'Topic similarity visualization.html'))

embeddings = topic_model_challenge.embedding_model.embed_documents(docs)
fig = topic_model_challenge.visualize_documents(docs, embeddings=embeddings)
fig.write_html(os.path.join(path_topic, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
topics_new_challenge = topic_model_challenge.reduce_outliers(docs, topics_challenge, probabilities=probs_challenge, strategy="probabilities")

# persist the document topics
for index, topic in zip(indice, topics_new_challenge):
    df.at[index, 'Challenge_topic'] = topic

del df['Challenge_original_content']
del df['Challenge_preprocessed_content']
del df['Challenge_gpt_summary_preprocessed_content']

df.to_json(os.path.join(path_output, 'topics.json'), indent=4, orient='records')
