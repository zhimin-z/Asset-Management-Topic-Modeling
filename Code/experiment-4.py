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

os.environ["WANDB_API_KEY"] = '9963fa73f81aa361bdbaf545857e1230fc74094c'
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "150"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

wandb_project = 'asset-management-project'
wandb.login()

path_dataset = 'Dataset'
df_all = pd.read_json(os.path.join(path_dataset, 'all_filtered.json'))
docs = df_all[df_all['Challenge_original_content'] != '']['Challenge_original_content'].tolist()

# set general sweep configuration
sweep_configuration = {
    "name": "experiment-1",
    "metric": {
        'name': 'CoherenceCV',
        'goal': 'maximize'
    },
    "method": "grid",
    "parameters": {
        'n_neighbors': {
            'values': list(range(10, 110, 10))
        },
        'n_components': {
            'values': list(range(2, 12, 2))
        },
        'ngram_range': {
            'values': list(range(3, 6))
        },
    }
}

# set default sweep configuration
config_defaults = {
    'model_name': 'all-mpnet-base-v2',
    'metric_distane': 'manhattan',
    'low_memory': False,
    'max_cluster_size': 800,
    'min_cluster_size': 15,
    'stop_words': 'english',
    'reduce_frequent_words': True
}


def train():
    # Initialize a new wandb run
    with wandb.init() as run:
        # update any values not set by sweep
        run.config.setdefaults(config_defaults)

        # Step 1 - Extract embeddings
        embedding_model = SentenceTransformer(run.config.model_name)

        # Step 2 - Reduce dimensionality
        umap_model = UMAP(n_neighbors=wandb.config.n_neighbors, n_components=wandb.config.n_components,
                          metric=run.config.metric_distane, low_memory=run.config.low_memory)

        # Step 3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN()

        # Step 4 - Tokenize topics
        vectorizer_model = TfidfVectorizer(
            stop_words=run.config.stop_words, ngram_range=(1, wandb.config.ngram_range))

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
            # Step 7 - Track model stages
            # verbose=True
        )

        topic_model.fit_transform(docs)

        # Preprocess documents
        cleaned_docs = topic_model._preprocess_text(docs)
    
        # Extract vectorizer and tokenizer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer_model.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        # Create topic words
        topic_words = [[dictionary.token2id[w] for w in words if w in dictionary.token2id] for _ in range(topic_model.nr_topics)]

        # this creates a list of the token ids (in the format of integers) of the words in words that are also present in the 
        # dictionary created from the preprocessed text. The topic_words list contains list of token ids for each topic.

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


sweep_id = wandb.sweep(sweep_configuration, project=wandb_project)
# Create sweep with ID: j7pnz7gn
wandb.agent(sweep_id=sweep_id, function=train)
