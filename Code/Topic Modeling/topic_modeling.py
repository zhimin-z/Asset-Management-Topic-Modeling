from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

import gensim.corpora as corpora
import pandas as pd
import wandb
import os

path_result = 'Result'
path_dataset = 'Dataset'

path_general = os.path.join(path_result, 'General')

path_model = os.path.join(path_general, 'Model')
if not os.path.exists(path_model):
    os.makedirs(path_model)

wandb_project = 'asset-management-topic-modeling'

os.environ["WANDB_API_KEY"] = '9963fa73f81aa361bdbaf545857e1230fc74094c'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "100"

# set default sweep configuration
config_defaults = {
    # Refer to https://www.sbert.net/docs/pretrained_models.html
    'model_name': 'all-mpnet-base-v2',
    'metric_distane': 'manhattan',
    'calculate_probabilities': True,
    'reduce_frequent_words': True,
    'prediction_data': True,
    'low_memory': False,
    'min_cluster_size': 20,
    'random_state': 42,
    'n_components': 5,
    'ngram_range': 2
}

config_sweep = {
    "method": "grid",
    "metric": {
        'name': 'Coherence CV',
        'goal': 'maximize'
    },
}


class TopicModeling:
    def __init__(self, docs_name):
        # Initialize an empty list to store top models
        self.top_models = []
        self.path_model = path_model
        
        df = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))
        self.docs = df[df[docs_name].notna()][docs_name].tolist()
        
        if 'Solution' in docs_name:
            config_defaults['min_cluster_size'] = 5
        
        config_sweep['name'] = docs_name
        config_sweep['parameters'] = {
            'min_samples': {
                'values': list(range(1, config_defaults['min_cluster_size'] + 1))
                },
            }
        
    def __train(self):
        # Initialize a new wandb run
        with wandb.init() as run:
            # update any values not set by sweep
            run.config.setdefaults(config_defaults)

            # Step 1 - Extract embeddings
            embedding_model = SentenceTransformer(run.config.model_name)

            # Step 2 - Reduce dimensionality
            umap_model = UMAP(n_components=run.config.n_components, metric=run.config.metric_distane,
                              random_state=run.config.random_state, low_memory=run.config.low_memory)

            # Step 3 - Cluster reduced embeddings
            hdbscan_model = HDBSCAN(min_cluster_size=run.config.min_cluster_size,
                                    min_samples=wandb.config.min_samples, prediction_data=run.config.prediction_data)

            # Step 4 - Tokenize topics
            vectorizer_model = TfidfVectorizer(ngram_range=(1, run.config.ngram_range))

            # Step 5 - Create topic representation
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=run.config.reduce_frequent_words)

            # Step 6 - Fine-tune topic representation
            representation_model = KeyBERTInspired()

            # All steps together
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=representation_model,
                calculate_probabilities=run.config.calculate_probabilities
            )

            topics, _ = topic_model.fit_transform(self.docs)

            # Preprocess Documents
            documents = pd.DataFrame({"Document": self.docs,
                                      "ID": range(len(self.docs)),
                                      "Topic": topics})
            documents_per_topic = documents.groupby(
                ['Topic'], as_index=False).agg({'Document': ' '.join})
            cleaned_docs = topic_model._preprocess_text(
                documents_per_topic.Document.values)

            # Extract vectorizer and analyzer from BERTopic
            vectorizer = topic_model.vectorizer_model
            analyzer = vectorizer.build_analyzer()

            # Extract features for Topic Coherence evaluation
            tokens = [analyzer(doc) for doc in cleaned_docs]
            dictionary = corpora.Dictionary(tokens)
            corpus = [dictionary.doc2bow(token) for token in tokens]
            topic_words = [[words for words, _ in topic_model.get_topic(
                topic)] for topic in range(len(set(topics))-1)]

            coherence_cv = CoherenceModel(
                topics=topic_words,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_v'
            )

            coherence_umass = CoherenceModel(
                topics=topic_words,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence='u_mass'
            )

            coherence_cuci = CoherenceModel(
                topics=topic_words,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_uci'
            )

            coherence_cnpmi = CoherenceModel(
                topics=topic_words,
                texts=tokens,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_npmi'
            )

            coherence_cv = coherence_cv.get_coherence()
            wandb.log({'Coherence CV': coherence_cv})
            wandb.log({'Coherence UMASS': coherence_umass.get_coherence()})
            wandb.log({'Coherence UCI': coherence_cuci.get_coherence()})
            wandb.log({'Coherence NPMI': coherence_cnpmi.get_coherence()})
            number_topics = topic_model.get_topic_info().shape[0] - 1
            wandb.log({'Topic Number': number_topics})
            wandb.log(
                {'Uncategorized Post Number': topic_model.get_topic_info().at[0, 'Count']})

            # persist top 5 topic models

            model_name = f'{config_sweep["name"]}_{run.id}'
            model = {
                'model_path': os.path.join(self.path_model, model_name),
                'model_metrics': {
                    'number_topics': number_topics,
                    'coherence_cv': coherence_cv,
                }
            }
            
            # Inspect the new model only the topic number is above 20 
            if number_topics > 20:
                # Add the new model to the list if model number is below 5
                if len(self.top_models) < 5:
                    self.top_models.append(model)
                    topic_model.save(model['model_path'])
                else:
                    # Find the model with the lowest topic number in the list
                    lowest_number_topics_model = min(
                        self.top_models, key=lambda x: x['model_metrics']['number_topics'])
                    if coherence_cv > lowest_number_topics_model['model_metrics']['coherence_cv']:
                        # Replace the model with the lowest topic number in the list with the new model
                        self.top_models.remove(lowest_number_topics_model)
                        os.remove(lowest_number_topics_model['model_path'])
                        self.top_models.append(model)
                        topic_model.save(model['model_path'])

    def sweep(self):
        wandb.login()
        sweep_id = wandb.sweep(config_sweep, project=wandb_project)
        wandb.agent(sweep_id, function=self.__train)
