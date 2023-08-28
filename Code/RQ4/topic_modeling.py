import gensim.corpora as corpora
import pandas as pd
import wandb
import os

# from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

path_output = os.path.join(os.getcwd(), 'Result', 'RQ4')
path_model = os.path.join(path_output, 'Model')
if not os.path.exists(path_model):
    os.makedirs(path_model)

wandb_project = 'asset-management-topic-modeling'

os.environ["WANDB_API_KEY"] = '9963fa73f81aa361bdbaf545857e1230fc74094c'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "100"

# set default sweep configuration
config_defaults = {
    # Refer to https://www.sbert.net/docs/pretrained_models.html
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'metric_distane': 'cosine',
    'calculate_probabilities': True,
    'reduce_frequent_words': True,
    'prediction_data': True,
    'low_memory': False,
    'random_state': 42,
    'ngram_range': 2,
}

config_sweep = {
    'method': 'grid',
    'metric': {
        'name': 'Coherence CV',
        'goal': 'maximize'
    },
    'parameters': {
        'n_components': {
            'values': list(range(3,8)),
        },
    }
}


class TopicModeling:
    def __init__(self, column_name, min_cluster_size=20):
        self.top_models = []
        self.path_model = path_model
        
        df = pd.read_json(os.path.join(path_output, 'labels.json'))
        self.docs = df[(df[column_name] != 'na') & (df[column_name] != 'non-issue') & (df[column_name] != 'intermittent issue')][column_name].tolist()
        
        config_defaults['min_cluster_size'] = min_cluster_size
        config_sweep['name'] = column_name
        config_sweep['parameters']['min_samples'] = {
            'values': list(range(1, config_defaults['min_cluster_size'] + 1))
        }
        
    def __train(self):
        # Initialize a new wandb run
        with wandb.init() as run:
            # update any values not set by sweep
            run.config.setdefaults(config_defaults)
            
            # Step 1 - Extract embeddings
            embedding_model = SentenceTransformer(run.config.model_name)

            # Step 2 - Reduce dimensionality
            umap_model = UMAP(n_components=wandb.config.n_components, metric=run.config.metric_distane,
                              random_state=run.config.random_state, low_memory=run.config.low_memory)

            # Step 3 - Cluster reduced embeddings
            hdbscan_model = HDBSCAN(min_cluster_size=run.config.min_cluster_size,
                                    min_samples=wandb.config.min_samples, prediction_data=run.config.prediction_data)

            # Step 4 - Tokenize topics
            # vectorizer_model = CountVectorizer(ngram_range=(1, run.config.ngram_range), stop_words='english')

            # Step 5 - Create topic representation
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=run.config.reduce_frequent_words)

            # # Step 6 - Fine-tune topic representation
            representation_model = KeyBERTInspired()

            # All steps together
            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                # vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                representation_model=representation_model,
                n_gram_range=(1, run.config.ngram_range),
                calculate_probabilities=run.config.calculate_probabilities
            )

            topic_model = topic_model.fit(self.docs)
            # topic_model.reduce_topics(self.docs, nr_topics='auto')

            # Preprocess Documents
            documents = pd.DataFrame({"Document": self.docs,
                                      "ID": range(len(self.docs)),
                                      "Topic": topic_model.topics_})
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
                topic)] for topic in range(len(set(topic_model.topics_))-1)]

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

            wandb.log({'Coherence CV': coherence_cv.get_coherence()})
            wandb.log({'Coherence UMASS': coherence_umass.get_coherence()})
            wandb.log({'Coherence UCI': coherence_cuci.get_coherence()})
            wandb.log({'Coherence NPMI': coherence_cnpmi.get_coherence()})
            wandb.log({'Topic Number': topic_model.get_topic_info().shape[0] - 1})
            wandb.log({'Uncategorized Post Number': topic_model.get_topic_info().at[0, 'Count']})

            model_name = f'{config_sweep["name"]}_{run.id}'
            topic_model.save(os.path.join(self.path_model, model_name), serialization='safetensors', save_ctfidf=True)


    def sweep(self):
        wandb.login()
        sweep_id = wandb.sweep(config_sweep, project=wandb_project)
        wandb.agent(sweep_id, function=self.__train)