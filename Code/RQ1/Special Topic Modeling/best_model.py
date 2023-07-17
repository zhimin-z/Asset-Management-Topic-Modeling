import os
import pickle
import pandas as pd

from bertopic import BERTopic

path_output = os.path.join(os.getcwd(), 'Result', 'RQ1', 'Special Topics')
path_topic = os.path.join(os.getcwd(), 'Code', 'RQ1', 'Special Topic Modeling')
path_model = os.path.join(path_topic, 'Model')
path_anomaly = os.path.join(path_topic, 'Anomaly')
path_root_cause = os.path.join(path_topic, 'Root Cause')
path_solution = os.path.join(path_topic, 'Solution')

name_model_anomaly = 'anomaly_qvzif0vx'
name_model_root_cause = 'anomaly_qvzif0vx'
name_model_solution = 'solution_48eniqrs'

df = pd.read_json(os.path.join(path_output, 'labels.json'))

df['Challenge_card_sorting_topic'] = -1
df['Challenge_root_cause_card_sorting_topic'] = -1
df['Solution_card_sorting_topic'] = -1

docs_anomaly = []
docs_root_cause = []
docs_solution = []

indice_anomaly = []
indice_root_cause = []
indice_solution = []

for index, row in df.iterrows():
    if row['Challenge_type'] == 'anomaly':
        indice_anomaly.append(index)
        docs_anomaly.append(row['Challenge_summary'])
        if row['Challenge_root_cause'] != 'na':
            indice_root_cause.append(index)
            docs_root_cause.append(row['Challenge_root_cause'])
    if row['Challenge_solution'] != 'na':
        indice_solution.append(index)
        docs_solution.append(row['Challenge_solution'])
        
for docs, indice, path, name, column in zip([docs_anomaly, docs_root_cause, docs_solution], [indice_anomaly, indice_root_cause, indice_solution], [path_anomaly, path_root_cause, path_solution], [name_model_anomaly, name_model_root_cause, name_model_solution], ['Challenge_card_sorting_topic', 'Challenge_root_cause_card_sorting_topic', 'Solution_card_sorting_topic']):
    topic_model = BERTopic.load(os.path.join(path_model, name))
    topic_number = topic_model.get_topic_info().shape[0] - 1
    topics, probs = topic_model.transform(docs)
    
    # persist the topic terms
    with open(os.path.join(path, 'Topic terms.pickle'), 'wb') as handle:
        topic_terms = []
        for i in range(topic_number):
            topic_terms.append(topic_model.get_topic(i))
        pickle.dump(topic_terms, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # persist the topic visualization
    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(path, 'Topic visualization.html'))
    
    fig = topic_model.visualize_barchart(top_n_topics=topic_number, n_words=10)
    fig.write_html(os.path.join(path, 'Term visualization.html'))
    
    fig = topic_model.visualize_heatmap()
    fig.write_html(os.path.join(path, 'Topic similarity visualization.html'))
    
    embeddings = topic_model.embedding_model.embed_documents(docs)
    fig = topic_model.visualize_documents(docs, embeddings=embeddings)
    fig.write_html(os.path.join(path, 'Document visualization.html'))
    
    # This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
    topics_new = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")
    
    # persist the document topics
    for index, topic in zip(indice, topics_new):
        df.at[index, column] = topic

df.to_json(os.path.join(path_output, 'topics.json'), indent=4, orient='records')
