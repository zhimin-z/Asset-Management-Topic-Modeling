import os
import pickle
import pandas as pd

# from gensim.parsing.preprocessing import strip_punctuation
from bertopic import BERTopic

path_output = os.path.join(os.getcwd(), 'Result', 'RQ1')
path_model = os.path.join(path_output, 'Model')
path_anomaly = os.path.join(path_output, 'Anomaly')
path_root_cause = os.path.join(path_output, 'Root Cause')
path_solution = os.path.join(path_output, 'Solution')

model_anomaly = 'Challenge_summary_547wgzi6'
model_root_cause = 'Challenge_root_rause_summary_547wgzi6'
model_solution = 'Solution_summary_406onakc'

df = pd.read_json(os.path.join(path_output, 'labels.json'))
        
for path, model in zip([path_anomaly, path_root_cause, path_solution], [model_anomaly, model_root_cause, model_solution]):
    docs = []
    indice = []
    column = '_'.join(model.split('_')[:-1])
    
    for index, row in df.iterrows():
        if row[column] != 'na':
            docs.append(row[column])
            indice.append(index)
            
    topic_model = BERTopic.load(os.path.join(path_model, model))
    topics, probs = topic_model.transform(docs)
    topic_number = topic_model.get_topic_info().shape[0] - 1
    
    # This uses the soft-clustering as performed by HDBSCAN to find the best matching topic for each outlier document.
    topics_new = topic_model.reduce_outliers(docs, topics=topics, probabilities=probs, strategy="probabilities")

    column_topic = f'{column}_topic'
    df[column_topic] = -1
    # persist the document topics
    for index, topic in zip(indice, topics_new):
        df.at[index, column_topic] = topic
    
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
