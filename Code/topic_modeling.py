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

wandb_project = 'asset-management-topic-modeling'
sweep_count = 500

os.environ["WANDB_API_KEY"] = '9963fa73f81aa361bdbaf545857e1230fc74094c'
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = str(sweep_count)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# set default sweep configuration
config_defaults = {
    'model_name': 'all-mpnet-base-v2',
    'metric_distane': 'manhattan',
    'n_components': 5,
    'low_memory': False,
    'reduce_frequent_words': True,
}

# set general sweep configuration
sweep_defaults = {
    "metric": {
        'name': 'CoherenceCV',
        'goal': 'maximize'
    },
    "method": "grid",
}

config_challenges = {
    "parameters": {
        'min_samples_pct': {
            'values': [.1, .25, .5, .75, 1]
        },
        'ngram_range': {
            'values': list(range(1, 6))
        },
        'min_cluster_size': {
            'values': list(range(30, 101, 5))
        },
    },
}

config_solutions = {
    "parameters": {
        'min_samples_pct': {
            'values': [.1, .25, .5, .75, 1]
        },
        'ngram_range': {
            'values': list(range(1, 6))
        },
        'min_cluster_size': {
            'values': list(range(15, 101, 5))
        },
    },
}


class TopicModeling:
    def __init__(self, docs_name):
        self.sweep_defaults = sweep_defaults
        self.sweep_defaults['name'] = docs_name

        df_all = pd.read_json(os.path.join('Dataset', 'all_filtered.json'))
        if docs_name in ['Solution_original_content', 'Solution_original_content_gpt_summary', 'Solution_preprocessed_content']:
            df_all = df_all[df_all['Solution_original_content'].isnull()
                            == False]
            df_all = df_all[df_all['Solution_original_content'] != '']
            self.sweep_defaults.update(config_solutions)
        else:
            self.sweep_defaults.update(config_challenges)
        self.docs = df_all[docs_name].tolist()

    def __train(self):
        # Initialize a new wandb run
        with wandb.init() as run:
            # update any values not set by sweep
            run.config.setdefaults(config_defaults)

            # Step 1 - Extract embeddings
            embedding_model = SentenceTransformer(run.config.model_name)

            # Step 2 - Reduce dimensionality
            umap_model = UMAP(n_components=wandb.config.n_components, metric=run.config.metric_distane, low_memory=run.config.low_memory)

            # Step 3 - Cluster reduced embeddings
            min_samples = int(wandb.config.min_cluster_size *
                              wandb.config.min_samples_pct)
            min_samples = 5 if min_samples < 5 else min_samples
            hdbscan_model = HDBSCAN(
                min_cluster_size=wandb.config.min_cluster_size, min_samples=min_samples)

            # Step 4 - Tokenize topics
            vectorizer_model = TfidfVectorizer(ngram_range=(1, wandb.config.ngram_range))

            # Step 5 - Create topic representation
            ctfidf_model = ClassTfidfTransformer(
                reduce_frequent_words=run.config.reduce_frequent_words)

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
            topic_words = [[words for words, _ in topic_model.get_topic(topic)] for topic in range(len(set(topics))-1)]

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

            wandb.log({'CoherenceCV': coherence_cv.get_coherence()})
            wandb.log({'CoherenceUMASS': coherence_umass.get_coherence()})
            wandb.log({'CoherenceUCI': coherence_cuci.get_coherence()})
            wandb.log({'CoherenceNPMI': coherence_cnpmi.get_coherence()})
            wandb.log(
                {'Number of Topics': topic_model.get_topic_info().shape[0] - 1})
            wandb.log(
                {'Number of Uncategorized': topic_model.get_topic_info().at[0, 'Count']})

    def sweep(self):
        wandb.login()
        sweep_id = wandb.sweep(self.sweep_defaults, project=wandb_project)
        wandb.agent(sweep_id, function=self.__train)