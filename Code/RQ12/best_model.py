import os
import pickle
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

path_output = os.path.join('Result', 'RQ12')
path_model = os.path.join(path_output, 'Model')

model_name = 'Challenge_preprocessed_gpt_summary_3m4nndtv'
column = '_'.join(model_name.split('_')[:-1])

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
df = pd.read_json(os.path.join('Dataset', 'preprocessed.json'))
df['Challenge_topic'] = -1

indice = []
docs = []

for index, row in df.iterrows():
    if len(row[column]):
        indice.append(index)
        docs.append(row[column])

topic_model = BERTopic.load(os.path.join(path_model, model_name), embedding_model=embedding_model)
topic_number = topic_model.get_topic_info().shape[0] - 1
topics, probs = topic_model.transform(docs)
topics = topic_model.reduce_outliers(docs, topics)

# persist the document topics
for index, topic in zip(indice, topics):
    df.at[index, 'Challenge_topic'] = topic

# persist the topic terms
with open(os.path.join(path_output, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number):
        topic_terms.append(topic_model.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_output, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(top_n_topics=topic_number, n_words=10)
fig.write_html(os.path.join(path_output, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(path_output, 'Topic similarity visualization.html'))

df = df[df.columns.drop(list(df.filter(regex=r'preprocessed|gpt')))]
df.to_json(os.path.join(path_output, 'topics.json'), indent=4, orient='records')
