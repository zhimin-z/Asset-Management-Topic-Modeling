import os
import pickle
import pandas as pd

from bertopic import BERTopic
from wordcloud import WordCloud
from matplotlib import pyplot as plt

path_result = 'Result'
path_dataset = 'Dataset'

path_general = os.path.join(path_result, 'General')
path_model = os.path.join(path_general, 'Model')

path_challenge = os.path.join(path_result, 'Challenge')
# path_solution = os.path.join(path_result, 'Solution')

name_model_challenge = 'Challenge_gpt_summary_skk6x33z'
# name_model_solution = 'Solution_gpt_summary_jbgth36u'

df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

df['Challenge_topic'] = -1
# df['Solution_topic'] = -1

indice_challenge = []
# indice_solution = []

docs_challenge = []
# docs_solution = []

column_challenge = '_'.join(name_model_challenge.split('_')[:-1])
# column_solution = '_'.join(name_model_solution.split('_')[:-1])

for index, row in df.iterrows():
    if pd.notna(row[column_challenge]):
        indice_challenge.append(index)
        docs_challenge.append(row[column_challenge])
    # if pd.notna(row[column_solution]):
    #     indice_solution.append(index)
    #     docs_solution.append(row[column_solution])
        
topic_model_challenge = BERTopic.load(os.path.join(path_model, name_model_challenge))
# topic_model_solution = BERTopic.load(os.path.join(path_model, name_model_solution))

topic_number_challenge = topic_model_challenge.get_topic_info().shape[0] - 1
# topic_number_solution = topic_model_solution.get_topic_info().shape[0] - 1

topics_challenge, probs_challenge = topic_model_challenge.transform(docs_challenge)
# topics_solution, probs_solution = topic_model_solution.transform(docs_solution)

# persist the topic terms
with open(os.path.join(path_challenge, 'Topic terms.pickle'), 'wb') as handle:
    topic_terms = []
    for i in range(topic_number_challenge):
        topic_terms.append(topic_model_challenge.get_topic(i))
    pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(path_solution, 'Topic terms.pickle'), 'wb') as handle:
#     topic_terms = []
#     for i in range(topic_number_solution):
#         topic_terms.append(topic_model_solution.get_topic(i))
#     pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

fig = topic_model_challenge.visualize_topics()
fig.write_html(os.path.join(path_challenge, 'Topic visualization.html'))

# fig = topic_model_solution.visualize_topics()
# fig.write_html(os.path.join(path_solution, 'Topic visualization.html'))

fig = topic_model_challenge.visualize_barchart(top_n_topics=topic_number_challenge, n_words=10)
fig.write_html(os.path.join(path_challenge, 'Term visualization.html'))

# fig = topic_model_solution.visualize_barchart(top_n_topics=topic_number_solution, n_words=10)
# fig.write_html(os.path.join(path_solution, 'Term visualization.html'))

fig = topic_model_challenge.visualize_heatmap()
fig.write_html(os.path.join(path_challenge, 'Topic similarity visualization.html'))

# fig = topic_model_solution.visualize_heatmap()
# fig.write_html(os.path.join(path_solution, 'Topic similarity visualization.html'))

hierarchical_topics = topic_model_challenge.hierarchical_topics(docs_challenge)
fig = topic_model_challenge.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig.write_html(os.path.join(path_challenge, 'Hierarchical clustering visualization.html'))

# hierarchical_topics = topic_model_solution.hierarchical_topics(docs_solution)
# fig = topic_model_solution.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
# fig.write_html(os.path.join(path_solution, 'Hierarchical clustering visualization.html'))

embeddings = topic_model_challenge.embedding_model.embed_documents(docs_challenge)
fig = topic_model_challenge.visualize_documents(docs_challenge, embeddings=embeddings)
fig.write_html(os.path.join(path_challenge, 'Document visualization.html'))

# embeddings = topic_model_solution.embedding_model.embed_documents(docs_solution)
# fig = topic_model_solution.visualize_documents(docs_solution, embeddings=embeddings)
# fig.write_html(os.path.join(path_solution, 'Document visualization.html'))

# This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
topics_new_challenge = topic_model_challenge.reduce_outliers(docs_challenge, topics_challenge, probabilities=probs_challenge, strategy="probabilities")
# topics_new_solution = topic_model_solution.reduce_outliers(docs_solution, topics_solution, probabilities=probs_solution, strategy="probabilities")

path_wordcloud_challenge = os.path.join(path_challenge, 'Wordcloud')
# path_wordcloud_solution = os.path.join(path_solution, 'Wordcloud')

if not os.path.exists(path_wordcloud_challenge):
    os.makedirs(path_wordcloud_challenge)
    
# if not os.path.exists(path_wordcloud_solution):
#     os.makedirs(path_wordcloud_solution)

# Preprocess Documents
documents_challenge = pd.DataFrame({'Document': docs_challenge, 'Topic': topics_new_challenge})
# documents_solution = pd.DataFrame({'Document': docs_solution, 'Topic': topics_new_solution})

documents_per_topic_challenge = documents_challenge.groupby(['Topic']).agg({'Document': ' '.join}).reset_index()
# documents_per_topic_solution = documents_solution.groupby(['Topic']).agg({'Document': ' '.join}).reset_index()

for index, row in documents_per_topic_challenge.iterrows():
    wordcloud = WordCloud(width=1000, height=1000, background_color='white', min_font_size=10).generate(row['Document'])
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(path_wordcloud_challenge, f'Topic_{row["Topic"]}'+'.png'), bbox_inches='tight')
    plt.close()

# for index, row in documents_per_topic_solution.iterrows():
#     wordcloud = WordCloud(width=1000, height=1000, background_color='white', min_font_size=10).generate(row['Document'])
#     plt.figure(figsize=(10, 10), facecolor=None)
#     plt.imshow(wordcloud)
#     plt.axis("off")
#     plt.tight_layout(pad=0)
#     plt.savefig(os.path.join(path_wordcloud_solution, f'Topic_{row["Topic"]}'+'.png'), bbox_inches='tight')
#     plt.close()

# persist the document topics
for index, topic in zip(indice_challenge, topics_new_challenge):
    df.at[index, 'Challenge_topic'] = topic

# for index, topic in zip(indice_solution, topics_new_solution):
#     df.at[index, 'Solution_topic'] = topic

del df['Challenge_original_content']
del df['Challenge_preprocessed_content']
del df['Challenge_gpt_summary_preprocessed_content']

# del df['Solution_original_content']
# del df['Solution_preprocessed_content']
# del df['Solution_gpt_summary']

df.to_json(os.path.join(path_general, 'topics.json'), indent=4, orient='records')
