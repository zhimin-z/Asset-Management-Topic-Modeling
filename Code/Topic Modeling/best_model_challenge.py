
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
path_model = os.path.join(path_general, 'Model')

model_name = 'Challenge_gpt_summary_skk6x33z'

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

df['Challenge_topic'] = -1

indice = []
docs = []

column_challenge = '_'.join(model_name.split('_')[:-1])

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]):
        indice.append(index)
        docs.append(row[column_challenge])
        
topic_model = BERTopic.load(os.path.join(path_model, model_name))
topic_number = topic_model.get_topic_info().shape[0] - 1
topics, probs = topic_model.transform(docs)

# persist the topic terms
with open(os.path.join(path_challenge, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number):
        topic_terms.append(topic_model.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_challenge, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(top_n_topics=topic_number, n_words=10)
fig.write_html(os.path.join(path_challenge, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(
    path_challenge, 'Topic similarity visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(docs)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(
    path_challenge, 'Hierarchical clustering visualization.html'))

embeddings = topic_model.embedding_model.embed_documents(docs)
fig = topic_model.visualize_documents(docs, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
topics_new = topic_model.reduce_outliers(
    docs, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_challenge, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {'Document': docs, 'Topic': topics_new})
documents_per_topic = documents.groupby(['Topic']).agg(
    {'Document': ' '.join}).reset_index()

for index, row in documents_per_topic.iterrows():
    wordcloud = WordCloud(
        width=1000, height=1000, background_color='white', min_font_size=10).generate(row['Document'])
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud,
                f'Topic_{row["Topic"]}'+'.png'), bbox_inches='tight')
    plt.close()

# persist the document topics

for index, topic in zip(indice, topics_new):
    df.at[index, 'Challenge_topic'] = topic

del df['Challenge_original_content']
del df['Challenge_preprocessed_content']
del df['Challenge_gpt_summary']

df.to_json(os.path.join(path_general, 'topics.json'),
           indent=4, orient='records')
