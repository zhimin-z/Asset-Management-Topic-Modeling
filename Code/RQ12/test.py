import os
import re
import time
import math
import spacy
import pickle
import openai
import random
import textstat
import warnings
import itertools
import webbrowser
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from bs4 import BeautifulSoup
from plotly.subplots import make_subplots
from collections import namedtuple, Counter
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import kruskal, kendalltau, chi2_contingency
from gensim.parsing.preprocessing import remove_stopwords, strip_numeric, strip_punctuation, STOPWORDS

_here = os.path.dirname(os.path.abspath(__file__))
path_dataset = os.path.join(_here, '../../Dataset')
path_result = os.path.join(_here, '../../Result')

path_rq12 = os.path.join(path_result, 'RQ12')

random_state = 42

warnings.filterwarnings('ignore')

# import subprocess
# subprocess.run(['python', '-m' 'spacy', 'download', 'en_core_web_sm'])
nlp = spacy.load("en_core_web_sm")

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', None)

openai.api_key = os.getenv('OPENAI_API_KEY', 'xxxxxx')

prompt_summary = '''Refine the title of the following post to make it short and clear in simple English.\n###\n'''

tools_keyword_mapping = {
    'Aim': ['aim'],
    'Amazon SageMaker': ['amazon', 'aws', 'maker', 'sage'],
    'Azure Machine Learning': ['aml', 'azure', 'microsoft'],
    'ClearML': ['clearml'],
    'cnvrg.io': ['cnvrg'],
    'Codalab': ['codalab'],
    'Comet': ['comet'],
    'Determined': ['determined'],
    'Domino': ['domino'],
    'DVC': ['dvc'],
    'Guild AI': ['guild'],
    'H2O AI Cloud': ['h2o'],
    'Kedro': ['kedro'],
    'MLflow': ['databricks', 'mlflow'],
    'MLRun': ['mlrun'],
    'Neptune': ['neptune'],
    'Optuna': ['optuna'],
    'Polyaxon': ['polyaxon'],
    'Sacred': ['sacred'],
    'SigOpt': ['sigopt'],
    'Valohai': ['valohai'],
    'Verta': ['modeldb', 'verta'],
    'Vertex AI': ['google', 'gcp', 'vertex'],
    'Weights & Biases': ['biases', 'wandb', 'weights']
}

tools_keyword_set = set(itertools.chain(*tools_keyword_mapping.values()))

keyword_prefix_symbols = [
    ' ', 
    '`', 
    '\'',
    '"',
    '('
    '['
]

foundation_model_keywords = [
    'alpaca',
    'aquila',
    'baichuan',
    'baize',
    'bard',
    'bayling',
    'belle',
    'blip',
    'bliva',
    'bloom',
    'cabrita',
    'cama',
    'cerebras',
    'cheetor',
    'chimed',
    'chatglm',
    'chatgpt', 
    'chatrwkv',
    'chimera',
    'claude',
    'codegen',
    'codet5',
    'codellama',
    'codex',
    'code llama',
    'dataoptim-llava',
    'decicoder',
    'deepseek',
    'dlite',
    'dolly',
    'falcon',
    'fastchat',
    'flan',
    'foundation model', 
    'galactica',
    'galpaca',
    'gemini',
    'gpt', 
    'guanaco',
    'h2ogpt',
    'honeybee',
    'infmllm',
    'internlm',
    'instructblip',
    'koala',
    'large action model', 
    'large audio model',
    'large audio language model',
    'large audio-language model',
    'large language model', 
    'large multimodal language model', 
    'large multi-modal language model', 
    'large multimodal model', 
    'large multi-modal model', 
    'large speech model',
    'large time series model',
    'large trajectory model',
    'large vision language model', 
    'large vision model', 
    'large vision-language model', 
    'lalm',
    'lavin',
    'llama',
    'llava',
    'llm', 
    'lmm', 
    'lvis-instruct4v',
    'lvlm', 
    'lvm', 
    'minigpt',
    'mistral',
    'mllm',
    'mmicl',
    'monkey-chat',
    'moss',
    'mplug-owl',
    'mpt',
    'multimodal-gpt',
    'muffin',
    'openflamingo',
    'openhermes',
    'openllama',
    'openlm',
    'open-assistant',
    'orca',
    'otter',
    'palmyra',
    'palm-coder',
    'pandagpt',
    'pangu',
    'phi-',
    'phoenix',
    'puremm',
    'pythia',
    'qwen',
    'rbdash',
    'redpajama',
    'replit code',
    'santacoder',
    'sharegpt4v',
    'slm',
    'small language model',
    'stablelm',
    'stackllama',
    'starchat',
    'starcoder',
    'stablelm',
    't5',
    'tiny language model',
    'tlm',
    'transcore',
    'wizardcoder',
    'wizardlm',
    'ul2',
    'vicuna',
    'visualglm',
    'vpgtrans',
    'ultralm',
    'wemm',
    'xgen',
    'yalm'
]

tools_open_post = [
    'Domino',
    'DVC',
    'Guild AI',
    'MLflow',
    'SigOpt'
]

keywords_image = {
    ".jpg", 
    ".pdf", 
    ".jpeg", 
    ".gif", 
    ".bmp", 
    ".webp", 
    ".svg", 
    ".tiff"
}

keywords_patch = {
    'pull',
}

keywords_issue = {
    'answers',
    'discussions',
    'forums',
    'issues',
    'questions',
    'stackoverflow',
}

keywords_tool = {
    'github',
    'gitlab',
    'pypi',
}

keywords_doc = {
    'developers',
    'docs',
    'documentation',
    'features',
    'library',
    'org',
    'wiki',
}

keywords_tutorial = {
    'guide',
    'learn',
    'tutorial',
}

error_words_basics = {
    'break',
    'bug',
    'ceas',
    'clos',
    'crash',
    'defect',
    'error',
    'fail',
    'fault',
    'fix',
    'finish',
    'flaw',
    'glitch',
    'halt',
    'malfunction',
    'mistak',
    'paus',
    'shutdown',
    'stop',
    'terminat',
}

stop_words_se = {
    'ability',
    'abilities',
    'accident',
    'accidents',
    'acknowledgement',
    'action',
    'actions',
    'activities',
    'activity',
    'advantage',
    'advantages',
    'advice',
    'ai',
    'alternative',
    'alternatives',
    'announcement',
    'announcements',
    'anomaly'
    'anomalies'
    'answer',
    'answers',
    'appreciation',
    'approach',
    'approaches',
    'article',
    'articles',
    'assistance',
    'attempt',
    'author',
    'behavior',
    'behaviour',
    'benefit',
    'benefits',
    'bit',
    'bits',
    'block',
    'blocks',
    'blog',
    'blogs',
    'body',
    'building',
    'case',
    'cases',
    'categories',
    'categorization',
    'category',
    'cause',
    'causes',
    'challenge',
    'challenges',
    'change',
    'changes',
    # 'char',
    'check',
    'choice',
    'choices',
    'clarification',
    'code',
    'collection',
    'com',
    'combination',
    # 'commmunication',
    # 'community',
    # 'communities',
    # 'company',
    # 'companies',
    # 'computer',
    # 'computers',
    # 'concept',
    # 'concepts',
    'concern',
    'concerns',
    # 'condition',
    # 'conditions',
    'confirmation',
    'confusion',
    'consideration',
    # 'content',
    # 'contents',
    'context',
    # 'count',
    'couple',
    'couples',
    'course',
    'courses',
    'cross',
    # 'custom',
    'day',
    'days',
    'demand',
    'description',
    'descriptions',
    'desire',
    'detail',
    'details',
    'devops',
    'difference',
    'differences',
    'difficulties',
    'difficulty',
    'disadvantage',
    'disadvantages',
    'discrepancies',
    'discrepancy',
    'discussion',
    'discussions',
    'dislike',
    'distinction',
    'effect',
    'end',
    # 'engineering',
    'enquiries',
    'enquiry',
    'evidence',
    'example',
    'examples',
    'exception',
    'exceptions',
    'existence',
    'exit',
    'expectation',
    'experience',
    'expert',
    'experts',
    'explanation',
    'explanations',
    'fact',
    'facts',
    'favorite',
    'favorites',
    # 'feature',
    # 'features',
    'feedback',
    'feedbacks',
    'fix',
    'fixes',
    # 'float',
    # 'form',
    # 'forms',
    'forum'
    'forums'
    # 'functionality',
    # 'functionalities',
    'future',
    'goal',
    'goals',
    'guarantee',
    'guidance',
    'guideline',
    'guide',
    'guides',
    'guy',
    'guys',
    'harm',
    'hello',
    'help',
    'hour',
    'hours',
    'ibm',
    'idea',
    'ideas',
    'individual',
    'individuals',
    'info',
    'information',
    'inquiries',
    'inquiry',
    'insight',
    'instruction',
    'instructions',
    # 'int',
    'intelligence',
    'intent',
    'interest',
    'introduction',
    'investigation',
    'invitation',
    'ipynb',
    'issue',
    'issues',
    'kind',
    'kinds',
    'lack',
    'learning',
    'level',
    'levels',
    'look',
    'looks',
    'lot',
    'lots',
    'luck',
    'machine',
    'major',
    'manner',
    'manners',
    'manual',
    'mark',
    'means',
    'meaning',
    # 'message',
    # 'messages',
    'method',
    'methods',
    'ml',
    'mlops',
    'minute',
    'minutes',
    'month',
    'months',
    'need',
    'needs',
    'number',
    'numbers',
    'offer',
    'one',
    'ones',
    'opinion',
    'opinions',
    # 'org',
    # 'organization',
    # 'organizations',
    'outcome',
    'part',
    'parts',
    'past',
    'people',
    'person',
    'persons',
    'perspective',
    'perspectives',
    'place',
    'places',
    'point',
    'points',
    'post',
    'posts',
    'practice',
    'practices',
    'problem',
    'problems',
    'product',
    'products',
    'program',
    'programs',
    'project',
    'projects',
    'proposal',
    'proposals',
    'purpose',
    'purposes',
    'py',
    # 'python',
    'qa',
    'question',
    'questions',
    'reason',
    'reasons',
    # 'research',
    'result',
    'results',
    'scenario',
    'scenarios',
    'science',
    'screenshot',
    'screenshots',
    'second',
    'seconds',
    'section',
    'sense',
    'shortcoming',
    'shortcomings',
    'show',
    'shows',
    'situation',
    'software',
    'solution',
    'solutions',
    'start',
    # 'state',
    # 'statement',
    # 'states',
    # 'status',
    # 'step',
    # 'steps',
    # 'string',
    # 'student',
    # 'students',
    'study',
    'stuff',
    'success',
    'suggestion',
    'suggestions',
    'summary',
    'summaries',
    'surprise',
    'support',
    'supports',
    'talk',
    'talks',
    'task',
    'tasks',
    # 'teacher',
    # 'teachers',
    # 'team',
    # 'teams',
    'technique',
    'techniques',
    'technologies',
    'technology',
    'term',
    'terms',
    'tip',
    'tips',
    'thank',
    'thanks',
    'thing',
    'things',
    'thought',
    'thoughts',
    'three',
    'title',
    'today',
    'tomorrow',
    # 'tool',
    # 'tools',
    'total',
    'trouble',
    'troubles',
    'truth',
    'try',
    'tutorial',
    'tutorials',
    'two',
    'understand',
    'understanding',
    'usage',
    'use',
    'user',
    'users',
    'uses',
    # 'value',
    # 'values',
    'view',
    'viewpoint',
    'way',
    'ways',
    'weakness',
    'weaknesses',
    'week',
    'weeks',
    'word',
    'words',
    'work',
    'workaround',
    'workarounds',
    'works',
    'yeah',
    'year',
    'years',
    'yesterday',
}

stop_words_ml = {
    'ad',
    'ads',
    'advertisement',
    'advertisements',
    'algorithm',
    'algorithms',
    'analysis',
    'anomaly',
    'asr',
    'audio',
    'autoencoder',
    'automl',
    'autopilot',
    'bert',
    'bi',
    'chat',
    'chatbot',
    'classification',
    'classifier',
    'clustering',
    'cnn',
    'cv',
    'decision',
    'detection',
    'dimensionality',
    'embedding',
    'estimation',
    'extraction',
    'forecast',
    'forecaster',
    'forecasts',
    'forecasting',
    'forest',
    'fraud',
    'gan',
    'gesture',
    'gpt',
    'ica',
    'kmeans',
    'k-means',
    'knn',
    'language',
    'languages',
    'lda',
    'lstm',
    'mining',
    'modelling',
    'ngram',
    'n-gram',
    'nlp',
    'nmf',
    'ocr',
    'pca',
    'processing',
    'recognition',
    'recommendation',
    'recommendations',
    'recommender',
    'reduction',
    'regression',
    'regressor',
    'reinforcement',
    'rf',
    'rl',
    'rnn',
    'segmentation',
    'sentiment',
    'series',
    'sound',
    'spam',
    'speech',
    'stt',
    'strategy',
    'strategies',
    'svd',
    'svm',
    'text',
    'time',
    'timeseries'
    'topic',
    'topics',
    'transformer',
    'transformers',
    'translation',
    'translator',
    'tree',
    'trend',
    'trends',
    'tsne',
    'tts',
    'vae',
    'video',
    'vision',
    'voice',
}

stop_words = STOPWORDS.union(stop_words_se.union(stop_words_ml))

def count_code_line(block_list):
    total_loc = 0
    for block in block_list:
        for line in block.splitlines():
            if line.strip():
                total_loc += 1
    return total_loc

def extract_styles(content):
    soup = BeautifulSoup(content, 'html.parser')
    # extract links
    links = [a['href'] for a in soup.find_all('a', href=True)] 
    # extract code blocks type 1
    code_line1 = count_code_line([c.get_text() for c in soup.find_all('code')]) 
    # extract code blocks type 2
    code_line2 = count_code_line([c.get_text() for c in soup.find_all('blockquote')]) 
    # extract text
    for tag in soup.find_all(['code', 'blockquote', 'a']):
        tag.decompose()
    clean_text = soup.get_text(separator=' ')
    return clean_text, links, code_line1 + code_line2

def extract_code(content):
    code_patterns = [r'```.+?```', r'``.+?``', r'`.+?`']
    clean_text = content
    code_line = 0

    for code_pattern in code_patterns:
        code_snippets = re.findall(code_pattern, clean_text, flags=re.DOTALL)
        code_line += count_code_line(code_snippets)
        clean_text = re.sub(code_pattern, '', clean_text, flags=re.DOTALL)
    
    return clean_text, code_line

def extract_links(text):
    link_pattern1 = r"\!?\[.*?\]\((.*?)\)"
    links1 = re.findall(link_pattern1, text)
    clean_text = re.sub(link_pattern1, '', text)
    link_pattern2 = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    links2 = re.findall(link_pattern2, clean_text)
    clean_text = re.sub(link_pattern2, '', clean_text)
    links = links1 + links2
    return clean_text, links

def split_content(content):
    clean_text, links1, code_line1 = extract_styles(content)
    clean_text, code_line2 = extract_code(clean_text)
    clean_text, links2 = extract_links(clean_text)
    
    links = links1 + links2
    code_line = code_line1 + code_line2
    
    content_collection = namedtuple('Analyzer', ['text', 'links', 'code_line'])
    return content_collection(clean_text, links, code_line)

def word_frequency(text):
    word_counts = Counter(text.split())
    return word_counts

def extract_nouns_verbs(text, verb):
    doc = nlp(text)
    if verb:
        words = [token.text for token in doc if (token.pos_ == "VERB") or (token.pos_ == "NOUN")]
    else:
        words = [token.text for token in doc if token.pos_ == "NOUN"]
    return ' '.join(words)

def remove_words_with_substring(text, substring_list):
    words = text.split()
    for substring in substring_list:
        words = [word for word in words if substring not in word]
    return ' '.join(words)

def remove_escape_characters(s):
    return re.sub(r'\\.', ' ', s)

def preprocess_text(text, verb=True):
    clean_text = text.lower()
    clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
    clean_text = remove_escape_characters(clean_text)
    clean_text = strip_punctuation(clean_text)
    clean_text = strip_numeric(clean_text)
    clean_text = remove_words_with_substring(clean_text, tools_keyword_set)
    clean_text = remove_words_with_substring(clean_text, error_words_basics)
    clean_text = extract_nouns_verbs(clean_text, verb)
    clean_text = remove_stopwords(clean_text, stop_words)
    return clean_text

def analyze_links(links):
    total_links = len(links)
    image_links = 0
    documentation_links = 0
    tool_links = 0
    issue_links = 0
    patch_links = 0
    tutorial_links = 0
    example_links = 0
    
    for link in links:
        if any([image in link for image in keywords_image]):
            image_links += 1
        elif any([patch in link for patch in keywords_patch]):
            patch_links += 1
        elif any([issue in link for issue in keywords_issue]):
            issue_links += 1
        elif any([tool in link for tool in keywords_tool]):
            tool_links += 1
        elif any([doc in link for doc in keywords_doc]):
            documentation_links += 1
        elif any([tool in link for tool in keywords_tutorial]):
            tutorial_links += 1
        else:
            example_links += 1

    link_analysis = namedtuple('Analyzer', ['total', 'image', 'documentation', 'tool', 'issue', 'patch', 'tutorial', 'example'])
    return link_analysis(total_links, image_links, documentation_links, tool_links, issue_links, patch_links, tutorial_links, example_links)

def analyze_text(text):
    readability = textstat.flesch_reading_ease(text)
    reading_time = textstat.reading_time(text)
    
    text_analysis = namedtuple('Analyzer', ['readability', 'reading_time'])
    return text_analysis(readability, reading_time)

# expential backoff
def retry_with_backoff(fn, retries=2, backoff_in_seconds=1, *args, **kwargs):
    x = 0
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    while True:
        try:
            return fn(*args, **kwargs)
        except:
            if x == retries:
                raise

            sleep = backoff_in_seconds * 2 ** x + random.uniform(0)
            time.sleep(sleep)
            x += 1

def find_duplicates(in_list):  
    duplicates = []
    unique = set(in_list)
    for each in unique:
        count = in_list.count(each)
        if count > 1:
            duplicates.append(each)
    return duplicates

def input_challenge_type(link):
    print(link)
    
    choice_mapping = {
        "K": 'knowledge',
        "N": 'na',
        "P": 'problem',
    }
    valid_options = ["P", "K", "N"]
    
    webbrowser.open(link)
    choice = input("Choose an option - 'P', 'K', or 'N': ").upper()

    while choice not in valid_options:
        print("Invalid choice. Please choose 'P', 'K', or 'N' only.")
        choice = input("Choose an option - 'P', 'K', or 'N': ").upper()

    return choice_mapping[choice]

def filter_open_posts(df):
    df_copy = df.copy()
    for index, row in df_copy.iterrows():
        if ('Tool' in row['Platform']) and (row['Tools'][0] in tools_open_post):
            df_copy.drop(index, inplace=True)
    return df_copy

# Calculate Cramér's V
def CramersV_p(contingency_table):
    chi2, p, _, _ = chi2_contingency(contingency_table)
    n = np.sum(contingency_table)
    min_dim = min(contingency_table.shape)-1  # -1 for degrees of freedom
    Cramers_V = np.sqrt(chi2 / (n * min_dim))
    return Cramers_V, p

# Compute eta-squared
def etaSquared_p(resolved_time_list):
    H, p = kruskal(*resolved_time_list)
    # Get the sum of the lengths of all inner lists
    N = sum(len(inner_list) for inner_list in resolved_time_list)
    eta_squared = H / (N - 1)
    return eta_squared, p

def reshape_dataframe(df: pd.DataFrame, num_new_columns: int) -> pd.DataFrame:
    # Create a new index for reshaping
    df['new_index'] = df.index // num_new_columns

    # Create new column names
    df['col'] = df.index % num_new_columns

    # Pivot the dataframe
    wide_df = df.pivot(index='new_index', columns='col')

    # Sort column levels to ensure they are ordered as 'index', 'name' across each group
    sorted_columns = sorted(wide_df.columns, key=lambda x: (x[1], x[0]))
    wide_df = wide_df[sorted_columns]

    # Flatten the MultiIndex to get the desired format
    wide_df.columns = [f'{col[0]}_{col[1] + 1}' for col in wide_df.columns]
    wide_df = wide_df.reset_index(drop=True)

    return wide_df

topics = '''Topic 0: Pipeline Step -  A step in a pipeline that represents a specific task or action. 
Topic 1: Notebook Instance -  A managed environment for running Jupyter notebooks. 
Topic 2: Docker Image -  A lightweight, standalone, and executable software package that includes everything needed to run a piece of software. 
Topic 3: Version -  A specific release or iteration of a software or model. 
Topic 4: Log -  A record of events or actions that occur during the execution of a program or system. 
Topic 5: Plots -  Visual representations of data or information. 
Topic 6: Studio -  A platform or environment for developing, managing and deploying machine learning models. 
Topic 7: PyTorch -  An open-source machine learning framework for building and training neural networks. 
Topic 8: Prediction -  The process of using a trained model to make predictions or infer outcomes based on input data. 
Topic 9: Labeling Job -  A task or process of assigning labels or annotations to data for training or evaluation purposes. 
Topic 10: Hyperparameter Tuning -  The process of optimizing the hyperparameters of a machine learning model to improve its performance. 
Topic 11: TensorFlow -  An open-source machine learning framework for building and training neural networks. 
Topic 12: Spark -  An open source distributed computing system for big data processing and analytics. 
Topic 13: RStudio -  An integrated development environment (IDE) for the R programming language. 
Topic 14: Workspace -  A virtual environment or project space for organizing and managing files, code, and resources. 
Topic 15: Batch Transform -  The process of applying a trained model to a large batch of data for prediction or inference. 
Topic 16: Web Service -  A service or application accessible over the Internet or a network. 
Topic 17: Experiment -  A controlled test or investigation to gather data and evaluate hypotheses or models. 
Topic 18: Instance -  A virtual or physical computing resource used to run applications or processes. 
Topic 19: Sweep -  The process of systematically exploring a range of values or configurations for hyperparameters or other variables. 
Topic 20: Deploying Model -  The process of making a trained machine learning model available for use or consumption. 
Topic 21: Columns Values -  The values contained in specific columns or fields of a dataset or table. 
Topic 22: Inference Endpoint -  An endpoint or interface to make predictions or inferences using a deployed machine learning model. 
Topic 23: Compute -  The resources or infrastructure used for running computations or processes. 
Topic 24: Account -  A user account or profile used to access and manage resources or services. 
Topic 25: File Directory -  A location or folder in a file system that contains files and other directories. 
Topic 26: Object Attribute -  A characteristic or property of an object in object-oriented programming. 
Topic 27: Parameters Parameter -  Variables or values passed to a function or method to customize its behavior or operation. 
Topic 28: File Pandas -  The pandas library used for data manipulation and analysis. 
Topic 29: Git Repo -  A repository or version control system for managing and tracking changes to code and files. 
Topic 30: Training Job -  A task or process of training a machine learning model using labeled data. 
Topic 31: Save Model -  The process of persisting or storing a trained machine learning model for future use or deployment. 
Topic 32: Bucket -  A storage container or resource in cloud storage systems to store files and data. 
Topic 33: Limit Exceeded -  The condition where a resource or operation exceeds its allocated or allowed limit. 
Topic 34: Data Factory -  A platform or service for orchestrating and managing data pipelines and workflows. 
Topic 35: Format Convert -  The process of converting data or files from one format to another. 
Topic 36: Server -  A computer or system that provides resources or services to other computers or clients. 
Topic 37: Feature Store -  A centralized repository or system to store and manage machine learning features or data. 
Topic 38: Environment -  The configuration or setup of software, libraries, and dependencies required to run an application or process. 
Topic 39: Initialization Init -  The process of setting up or preparing a system, environment, or object for use. 
Topic 40: Custom Metrics -  User-defined or specific metrics used to evaluate or measure the performance of machine learning models. 
Topic 41: Compute Cluster -  A group or collection of interconnected computing resources or nodes used for parallel processing or distributed computing. 
Topic 42: Model Endpoint -  An interface or endpoint for accessing and utilizing a deployed machine learning model. 
Topic 43: Memory -  The storage or workspace used by a computer or system to store and manipulate data or instructions. 
Topic 44: Stuck Queue -  The condition where a job or task remains in a queue without progressing or completing. 
Topic 45: Distributed Training -  The process of training machine learning models using multiple computing resources or nodes in parallel. 
Topic 46: File Upload -  The process of transferring or sending files or data from a local system to a remote or cloud-based system. 
Topic 47: Python -  A popular programming language widely used in the MLOps workflow for developing machine learning models and data processing tasks. 
Topic 48: TensorBoard -  A web-based tool provided by TensorFlow to visualize and analyze machine learning models and experiments. 
Topic 49: Deploying Endpoint -  The process of making a machine learning model endpoint or API available for external access or consumption. 
Topic 50: Authentication -  The process of verifying the identity or credentials of a user or system to grant access to resources or services. 
Topic 51: Module Import -  The process of bringing external code, libraries, or modules into a program or environment for use. 
Topic 52: Deployment Model -  The process of deploying a trained machine learning model for production use or consumption. 
Topic 53: Artifact Download -  The process of retrieving or obtaining artifacts, such as models or data, from a storage or repository. 
Topic 54: File Read -  The process of accessing or extracting data or information from a file or document. 
Topic 55: Batch Prediction -  The process of making predictions or inferences on a batch or set of data using a trained machine learning model. 
Topic 56: API -  A set of rules and protocols that allows different software applications to communicate and interact with each other. 
Topic 57: API Key -  A unique identifier or token used to authenticate and authorize access to an API or service. 
Topic 58: File Storage -  The act or process of storing or saving data, files, or resources for future use or retrieval. 
Topic 59: Creating Terraform -  The process of creating or defining infrastructure resources using the Terraform infrastructure-as-code tool. 
Topic 60: CUDA Memory -  The memory space or resources used by NVIDIA GPUs for parallel computing and acceleration. 
Topic 61: Register Model -  The process of recording or adding a trained machine learning model to a registry or catalog for tracking and management. 
Topic 62: Log Metrics -  The process of recording performance metrics, such as accuracy or loss, during model training or evaluation. 
Topic 63: Config Configuration -  The settings, parameters, or options that define the behavior or operation of a system, application, or process. 
Topic 64: Inference Pipeline -  A sequence or series of steps or processes for making predictions or inferences using a deployed machine learning model. 
Topic 65: Invoke Endpoint -  The process of calling or triggering a deployed machine learning model endpoint to make predictions or inferences. 
Topic 66: Blob Storage -  A type of cloud storage that stores unstructured data as binary large objects (BLOBs). 
Topic 67: Comparison Performance -  The process of comparing or benchmarking the performance or capabilities of different machine learning models or algorithms. 
Topic 68: Loading Model -  The process of loading or importing a trained machine learning model into memory for use or further analysis. 
Topic 69: Pickle File -  A file format for serializing and deserializing Python objects, including machine learning models. 
Topic 70: Endpoint Lambda -  A serverless compute service provided by AWS to run code without provisioning or managing servers. 
Topic 71: File Download -  The process of retrieving or obtaining files from a remote or cloud-based source for use or analysis. 
Topic 72: Broken Link -  A hyperlink or URL that does not lead to the intended or expected destination or resource. 
Topic 73: Permissions Authorization -  The process of granting or restricting access to resources or services based on user roles or permissions. 
Topic 74: Exporting Data -  The process of saving or transferring data from one system or format to another for sharing or analysis. 
Topic 75: Datasets -  Collections or sets of data organized and structured for specific purposes or tasks. 
Topic 76: Model Registry -  A centralized repository or system for storing and managing machine learning models, metadata, and related artifacts. 
Topic 77: Image Read -  The process of reading or decoding JPEG image files to access or manipulate the image data. 
Topic 78: Artifacts -  Objects, files, or resources generated or produced during the machine learning workflow, such as trained models, reports, or visualizations. 
Topic 79: Deploy Model -  The process of making a trained machine learning model available for use or consumption. 
Topic 80: Endpoint Content -  The data or payload sent to or received from a machine learning model endpoint during prediction or inference. 
Topic 81: Creating Model -  The process of developing or building a machine learning model using data, algorithms, and techniques. 
Topic 82: Train Model -  The process of training a machine learning model using labeled data to learn patterns and make predictions or inferences. 
Topic 83: Score Model -  The process of applying a trained machine learning model to new or unseen data to generate predictions or inferences. 
Topic 84: Endpoint Ping -  The process of checking the availability or responsiveness of a machine learning model endpoint. 
Topic 85: Parallelization Job -  The process of parallelizing or distributing computational tasks or processes across multiple resources or nodes. 
Topic 86: Module Usage -  The process of utilizing libraries in software development to streamline coding and enhance modularity. 
Topic 87: Deployment Kubernetes -  The process of deploying machine learning models or applications using Kubernetes, an open-source container orchestration platform. 
Topic 88: Data Stores -  Storage systems or repositories to store and manage data used in machine learning workflows. 
Topic 89: Permission Denied -  The condition where access to a resource or operation is denied or restricted due to insufficient privileges or permissions. 
Topic 90: Training Reports -  Reports or documents generated or produced during machine learning model training, documenting the process, results, or insights. 
Topic 91: Log Model -  The process of recording or logging information, metrics, or events related to a machine learning model during training, evaluation, or deployment. 
Topic 92: Model Monitoring -  The process of continuously monitoring and tracking the performance, behavior, or usage of deployed machine learning models. 
Topic 93: Model Deployment -  The process of making a trained machine learning model available for use or consumption in production environments. 
Topic 94: Training Pipeline -  A step or component in a machine learning pipeline specifically designed for training models. 
Topic 95: Missing Module -  The condition where a required or expected module or library is not found or available in the system or environment. 
Topic 96: Install Package -  The process of installing or adding a software package or library to a system or environment for use. 
Topic 97: Logs CloudWatch -  The process of logging or capturing events, metrics, or information related to machine learning workflows on Amazon CloudWatch. 
Topic 98: Import Notebook -  The process of importing or loading a Jupyter notebook or notebook file into an environment or platform for execution or analysis. 
Topic 99: Training Model -  The process of training a machine learning model using labeled data to learn patterns and make predictions or inferences. 
Topic 100: Installation -  The process of setting up or configuring software, libraries, or dependencies on a system or environment. 
Topic 101: Quota Request -  The process of requesting an increase or adjustment to resource limits or quotas for specific services or operations. 
Topic 102: NEO Compiling -  The process of compiling or converting machine learning models into the NEO (Neural Efficient Object) format for optimized inference on edge devices. 
Topic 103: Connection -  The act or process of establishing a communication link or interface between different systems, components, or services. 
Topic 104: Training Container -  A containerized environment or runtime specifically designed for training machine learning models. 
Topic 105: Huggingface Model -  A machine learning model or framework provided by Hugging Face, specializing in natural language processing (NLP) tasks. 
Topic 106: Container Registry -  A repository or storage system for storing and managing container images used in the deployment of machine learning models or applications. 
Topic 107: Model ONNX -  The ONNX (Open Neural Network Exchange) format to represent and exchange machine learning models between different frameworks and platforms. 
Topic 108: Models -  Trained or pre-trained machine learning models used for making predictions or inferences on new or unseen data. 
Topic 109: Globals YAML -  A YAML (YAML Ain't Markup Language) file or configuration used to define global settings, parameters, or options for a system or environment. 
Topic 110: Deployment ACI -  The process of deploying machine learning models or applications using Azure Container Instances (ACI), a serverless compute service provided by Microsoft Azure. 
Topic 111: Trained Model -  A machine learning model that has undergone the training process using labeled data to learn patterns and make predictions or inferences. 
Topic 112: Web Interface -  A graphical user interface (GUI) or web-based interface for interacting with and managing machine learning models, data, or resources. 
Topic 113: scikit-learn -  A popular machine learning library used for data preprocessing, modeling, and evaluation. 
Topic 114: Dependency -  A software package, library, or module required by another software or system to function properly. 
Topic 115: Nested Runs -  The condition where multiple runs or executions overlap or are nested within each other in the MLOps workflow. 
Topic 116: Import Data -  The process of bringing external data or datasets into a system or environment for analysis, modeling, or processing. 
Topic 117: Endpoint Deploying -  The process of deploying or making a machine learning model endpoint or API available for external access or consumption. 
Topic 118: Endpoint Prediction -  The process of making predictions or inferences by calling or utilizing a deployed machine learning model endpoint. 
Topic 119: Lifecycle Configuration -  A configuration or set of instructions defining the lifecycle or behavior of a machine learning model or system. 
Topic 120: Output Inputs -  The results, outputs, or predictions generated by a machine learning model or system based on input data or parameters. 
Topic 121: Custom Job -  A customized or specific task or process designed for a particular machine learning workflow or requirement. 
Topic 122: Jupyter Notebook -  An open-source web-based interactive computing environment for creating and sharing documents containing live code, equations, visualizations and narrative text. 
Topic 123: Response Endpoint -  The output or response received from a machine learning model endpoint after making a prediction or inference request. 
Topic 124: Shared Cache -  A shared or common storage space or repository used by multiple processes or systems to store and retrieve data or resources. 
Topic 125: Training YOLO -  The process of training machine learning models using the YOLO (You Only Look Once) object detection framework. 
Topic 126: Checkpoints Training -  Intermediate or periodic snapshots or states of a machine learning model saved during the training process. 
Topic 127: Endpoint Transitioning -  The process or state of transitioning or changing the status or behavior of a machine learning model endpoint. 
Topic 128: Training File -  A file or dataset used to train machine learning models, typically containing labeled or annotated data. 
Topic 129: Inference -  The process of using a trained machine learning model to make predictions or inferences on new or unseen data. 
Topic 130: Connecting RDS -  The process of establishing a connection or interface between a machine learning system or application and a relational database management system (RDS). 
Topic 131: Model Serving -  The process of making trained machine learning models available for use or consumption in production environments. 
Topic 132: Tuning Model -  The process of optimizing or fine-tuning the hyperparameters or configurations of a machine learning model to improve its performance.'''

topic_list = []
topic_mapping = {}

for index, topic_entry in enumerate(topics.splitlines()):
    topic_name, topic_info = topic_entry.split(' - ')
    topic_name = topic_name.split(': ')[-1]
    topic = {
        'Index': '$C_{0' + str(index + 1) + '}$' if index < 9 else '$C_{' + str(index + 1) + '}$',
        'Topic': topic_name,
        'Description': topic_info,
    }
    topic_list.append(topic)
    topic_mapping[index] = topic_name
    
topic_df = pd.DataFrame(topic_list)
print(topic_df.to_latex(index=False, escape=True))

macro_topic2index_list = [
    ('Code Development', [26,27,56]),
    ('Code Management', [29]),
    ('Computation Management', [18,23,33,36,41,43,60,101]),
    # ('Cost Management', []),
    ('Data Development', [9,12,21,28]),
    ('Data Management', [32,34,37,53,66,74,75,77,78,88,116,124,130]),
    ('Environment Management', [1,2,3,6,13,14,38,39,47,51,59,63,86,87,95,96,98,100,104,106,109,114,122]),
    ('Experiment Management', [17,115]),
    ('File Management', [25,35,46,54,58,69,71,120,128]),
    ('Model Deployment', [8,16,20,22,42,49,52,55,65,70,79,83,93,110,117,118,129,131]),
    ('Model Development', [7,10,11,19,30,45,82,99,113,125,132]),
    ('Model Management', [31,61,68,76,81,102,105,107,108,111,126]),
    ('Network Management', [72,80,84,103,112,123,127]),
    ('Observability Management', [4,40,48,62,67,90,91,92,97]),
    ('Pipeline Management', [0,15,44,64,85,94,119,121]),
    # ('Quality Assurance Management', []),
    ('Security Management', [24,50,57,73,89]),
    ('User Interface Management', [5]),
]

topic_list = []
macro_topic_mapping = {}
macro_topic_indexing = {}
macro_topic2index_dict = {}
for index, topic_set in enumerate(macro_topic2index_list):
    macro_topic2index_dict[topic_set[0]] = topic_set[1]
    macro_topic_indexing[index] = topic_set[0]
    topic_list.extend(topic_set[1])
    for topic in topic_set[1]:
        macro_topic_mapping[topic] = index

print(find_duplicates(topic_list))
print(len(topic_df) == len(topic_list))
print(set(range(len(topic_list))).difference(set(range(topic_df.shape[0]))))

import pandas as pd
import plotly.graph_objects as go
import os

df = pd.read_json(os.path.join(path_rq12, 'macro-topics.json'))

categories = []
frequency_p = []
frequency_k = []

for index, group in df.groupby('Challenge_topic_macro'):
    categories.append(macro_topic_indexing[index])
    frequency_p.append(-len(group[group['Challenge_type'] == 'problem']))  # Note the negative value
    frequency_k.append(len(group[group['Challenge_type'] == 'knowledge']))

fig = go.Figure(data=[
    go.Bar(name='Knowledge', y=categories, x=frequency_k, text=frequency_k, textposition='outside', orientation='h'),
    go.Bar(name='Problem', y=categories, x=frequency_p, text=[-val for val in frequency_p], textposition='outside', orientation='h'),  # Negative values for display
])

tick_values = list(range(-1000, 1800, 500))  # Example tick intervals
tick_text = [str(abs(val)) for val in tick_values]

fig.update_layout(
    barmode='relative',  # This will place bars next to each other
    width=1200,  # Adjust width
    height=800,  # Adjust height
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(
        title="",
        zeroline=True, zerolinewidth=2, zerolinecolor='Black', tickvals=tick_values, ticktext=tick_text,  # Adjust tickvals and ticktext
        tickfont=dict(size=18),  # Enlarged x-axis tick font
        title_font=dict(size=20)  # Correct property name with underscore
    ),
    yaxis=dict(
        title="",
        autorange="reversed",
        tickfont=dict(size=18),  # Enlarged y-axis tick font
        title_font=dict(size=20)  # Correct property name with underscore
    ),
    legend=dict(
        font=dict(
            size=18  # Enlarged legend font size
        ),
        # Place the legend inside the chart area
        x=0.8,      # Position from the left (0-1)
        y=0.95,     # Position from the bottom (0-1)
    )
)

fig.update_traces(textfont_size=16)  # increase number as needed

fig.show()

fig.write_image(os.path.join(path_rq12, "AM-Macro-topics-frequency-histogram.pdf"))
