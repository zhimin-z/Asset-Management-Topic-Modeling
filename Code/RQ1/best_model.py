import os
import pickle
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

path_rq1 = os.path.join('Result', 'RQ1')
path_model = os.path.join(path_rq1, 'Model')

model_name = 'Challenge_preprocessed_title_4z17fib1'
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
new_topics = topic_model.reduce_outliers(docs, topics, strategy="distribution")

# persist the topic terms
with open(os.path.join(path_rq1, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number):
        topic_terms.append(topic_model.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_rq1, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(top_n_topics=topic_number, n_words=10)
fig.write_html(os.path.join(path_rq1, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(path_rq1, 'Topic similarity visualization.html'))

# persist the document topics
for index, topic in zip(indice, new_topics):
    df.at[index, 'Challenge_topic'] = topic

df = df[df.columns.drop(list(df.filter(regex=r'preprocessed|gpt')))]
df.to_json(os.path.join(path_rq1, 'topics.json'), indent=4, orient='records')
