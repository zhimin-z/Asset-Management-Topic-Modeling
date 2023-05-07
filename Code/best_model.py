
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
path_model = os.path.join(path_general, 'Model')

model_name = 'preprocessed_content_j7vecmqw'

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

column_challenge = 'Challenge_' + '_'.join(model_name.split('_')[:-1])
column_solution = 'Solution_' + '_'.join(model_name.split('_')[:-1])

df['Challenge_topic'] = -1
df['Solution_topic'] = -1

indice_challenge = []
indice_solution = []
docs_challenge = []
docs_solution = []

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]):
        indice_challenge.append(index)
        docs_challenge.append(row[column_challenge])
    if pd.notna(row[column_solution]):
        indice_solution.append(index)
        docs_solution.append(row[column_solution])
        
docs = docs_challenge + docs_solution
topic_model = BERTopic.load(os.path.join(path_model, model_name))
topics, probs = topic_model.transform(docs)

df_topics = topic_model.get_topic_info()
df_topics.to_json(os.path.join(
    path_general, 'Topic information.json'), indent=4, orient='records')

# persist the topic terms
with open(os.path.join(path_general, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(df_topics.shape[0] - 1):
        topic_terms.append(topic_model.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model.visualize_topics()
fig.write_html(os.path.join(path_general, 'Topic visualization.html'))

fig = topic_model.visualize_barchart(
    top_n_topics=df_topics.shape[0]-1, n_words=10)
fig.write_html(os.path.join(path_general, 'Term visualization.html'))

fig = topic_model.visualize_heatmap()
fig.write_html(os.path.join(
    path_general, 'Topic similarity visualization.html'))

fig = topic_model.visualize_term_rank()
fig.write_html(os.path.join(
    path_general, 'Term score decline visualization.html'))

hierarchical_topics = topic_model.hierarchical_topics(docs)
fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(
    path_general, 'Hierarchical clustering visualization.html'))

embeddings = topic_model.embedding_model.embed_documents(docs)
fig = topic_model.visualize_documents(docs, embeddings=embeddings)
fig.write_html(os.path.join(path_general, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
topics_new = topic_model.reduce_outliers(
    docs_challenge, topics, probabilities=probs, strategy="probabilities")

path_wordcloud = os.path.join(path_general, 'Wordcloud')
if not os.path.exists(path_wordcloud):
    os.makedirs(path_wordcloud)

# Preprocess Documents
documents = pd.DataFrame(
    {'Document': docs, 'Topic': topics_new})
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
        
topics_challenge, probs_challenge = topics[:len(docs_challenge)], probs[:len(docs_challenge)]
topics_solution, probs_solution = topics[len(docs_challenge):], probs[len(docs_challenge):]

# persist the document topics

for index, topic in zip(indice_challenge, topics_new[:len(docs_challenge)]):
    df.at[index, 'Challenge_topic'] = topic

for index, topic in zip(indice_solution, topics_new[len(docs_challenge):]):
    df.at[index, 'Solution_topic'] = topic

# del df['Challenge_original_content']
# del df['Challenge_preprocessed_content']
# del df['Challenge_gpt_summary']

# del df['Solution_original_content']
# del df['Solution_preprocessed_content']
# del df['Solution_gpt_summary']

df.to_json(os.path.join(path_general, 'topics.json'),
           indent=4, orient='records')
