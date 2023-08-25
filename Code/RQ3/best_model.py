import os
import pickle
import pandas as pd

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

path_output = os.path.join(os.getcwd(), 'Result', 'RQ3')
path_model = os.path.join(path_output, 'Model')
model_output = 'Resolution_summary_fi1j3qkz'

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
df = pd.read_json(os.path.join(path_output, 'labels.json'))
        
for path, model in zip([path_output], [model_output]):
    docs = []
    indice = []
    column = '_'.join(model.split('_')[:-1])
    
    for index, row in df.iterrows():
        if row[column] != 'na':
            docs.append(row[column])
            indice.append(index)
            
    topic_model = BERTopic.load(os.path.join(path_model, model), embedding_model=embedding_model)
    topic_number = topic_model.get_topic_info().shape[0] - 1
    topics, probs = topic_model.transform(docs)
    topics = topic_model.reduce_outliers(docs, topics)

    column_topic = f'{column}_topic'
    df[column_topic] = -1
    # persist the document topics
    for index, topic in zip(indice, topics):
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
