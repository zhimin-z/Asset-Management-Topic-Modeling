import os
import pickle
import pandas as pd

from bertopic import BERTopic

path_output = os.path.join(os.getcwd(), 'Result', 'RQ1', 'Special Topics')
path_topic = os.path.join(os.getcwd(), 'Code', 'RQ1', 'Special Topic Modeling')
path_model = os.path.join(path_topic, 'Model')
path_root_cause = path_anomaly = os.path.join(path_topic, 'Anomaly')
path_solution = os.path.join(path_topic, 'Solution')

name_model_root_cause = name_model_anomaly = 'anomaly_5b4iugcc'
name_model_solution = 'solution_nr6cyc9j'

df = pd.read_json(os.path.join(path_output, 'preprocessed.json'))

column_anomaly = 'Challenge_summary_topic'
column_root_cause = 'Challenge_root_cause_topic'
column_solution = 'Solution_topic'

df[column_anomaly] = -1
df[column_root_cause] = -1
df[column_solution] = -1

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
    if row['Solution'] != 'na':
        indice_solution.append(index)
        docs_solution.append(row['Solution'])
        
for docs, indice, path, name, column in zip([docs_anomaly, docs_root_cause, docs_solution], [indice_anomaly, indice_root_cause, indice_solution], [path_anomaly, path_root_cause, path_solution], [name_model_anomaly, name_model_root_cause, name_model_solution], [column_anomaly, column_root_cause, column_solution]):
    topic_model = BERTopic.load(os.path.join(path_model, name))
    
    topics, probs = topic_model.transform(docs)
    # This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
    topics_new = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")
    
    # persist the document topics
    for index, topic in zip(indice, topics_new):
        df.at[index, column] = topic
    
    if column == column_root_cause:
        continue
    
    topic_number = topic_model.get_topic_info().shape[0] - 1
    
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

df.to_json(os.path.join(path_output, 'topics.json'), indent=4, orient='records')
