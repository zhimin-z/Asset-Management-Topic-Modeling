import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np

from matplotlib import pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns",
              None, 'display.max_colwidth', None)

import subprocess

# for complex commands, with many args, use string + `shell=True`:
cmd_str = "!python -m spacy download en_core_web_sm -q"
subprocess.run(cmd_str, shell=True)

path_dataset = 'Dataset'

path_result = 'Result'
if not os.path.exists(path_result):
    os.makedirs(path_result)

path_general = os.path.join(path_result, 'General')
if not os.path.exists(path_general):
    os.makedirs(path_general)
    
# combine issues and questions

import re
import spacy

# Refer to https://textacy.readthedocs.io/en/stable/api_reference/text_stats.html
from textacy import text_stats

nlp = spacy.load('en_core_web_sm')
link_pattern = '(?P<url>ftp|https?://[^\s]+)'

df_issues = pd.read_json(os.path.join(path_dataset, 'issues.json'))
df_questions = pd.read_json(os.path.join(path_dataset, 'questions.json'))

df_issues['Challenge_link'] = df_issues['Issue_link']
df_issues['Challenge_original_content'] = df_issues['Issue_original_content']
df_issues['Challenge_preprocessed_content'] = df_issues['Issue_preprocessed_content']
df_issues['Challenge_summary'] = df_issues['Issue_gpt_summary']
df_issues['Challenge_creation_time'] = df_issues['Issue_creation_time']
df_issues['Challenge_answer_count'] = df_issues['Issue_answer_count']
df_issues['Challenge_score'] = df_issues['Issue_upvote_count'] - \
    df_issues['Issue_downvote_count']
df_issues['Challenge_closed_time'] = df_issues['Issue_closed_time']

challenge_content = df_issues['Issue_title'] + \
    '. ' + df_issues['Issue_body'].astype(str)
df_issues['Challenge_word_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_words(nlp(x)))
df_issues['Challenge_unique_word_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_unique_words(nlp(x)))
df_issues['Challenge_sentence_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_sents(nlp(x)))
df_issues['Challenge_information_entropy'] = challenge_content.apply(
    lambda x: text_stats.basics.entropy(nlp(x)))
df_issues['Challenge_readability'] = challenge_content.apply(
    lambda x: text_stats.readability.automated_readability_index(nlp(x)))
df_issues['Challenge_link_count'] = challenge_content.apply(
    lambda x: len(re.findall(link_pattern, x)))

df_issues['Solution_original_content'] = df_issues['Answer_original_content']
df_issues['Solution_preprocessed_content'] = df_issues['Answer_preprocessed_content']
df_issues['Solution_gpt_summary'] = df_issues['Answer_gpt_summary']

solution_content = df_issues['Answer_list'].apply(lambda x: '. '.join(x))
df_issues['Solution_word_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_words(nlp(x)))
df_issues['Solution_unique_word_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_unique_words(nlp(x)))
df_issues['Solution_sentence_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_sents(nlp(x)))
df_issues['Solution_information_entropy'] = solution_content.apply(
    lambda x: text_stats.basics.entropy(nlp(x)))
df_issues['Solution_readability'] = solution_content.apply(
    lambda x: text_stats.readability.automated_readability_index(nlp(x)) if x else pd.NA)
df_issues['Solution_link_count'] = solution_content.apply(
    lambda x: len(re.findall(link_pattern, x)))

df_questions['Challenge_link'] = df_questions['Question_link']
df_questions['Challenge_original_content'] = df_questions['Question_original_content']
df_questions['Challenge_preprocessed_content'] = df_questions['Question_preprocessed_content']
df_questions['Challenge_summary'] = df_questions['Question_gpt_summary']
df_questions['Challenge_creation_time'] = df_questions['Question_creation_time']
df_questions['Challenge_answer_count'] = df_questions['Question_answer_count']
df_questions['Challenge_comment_count'] = df_questions['Question_comment_count']
df_questions['Challenge_score'] = df_questions['Question_score']
df_questions['Challenge_closed_time'] = df_questions['Question_closed_time']
df_questions['Challenge_favorite_count'] = df_questions['Question_favorite_count']
df_questions['Challenge_last_edit_time'] = df_questions['Question_last_edit_time']
df_questions['Challenge_view_count'] = df_questions['Question_view_count']
df_questions['Challenge_follower_count'] = df_questions['Question_follower_count']
df_questions['Challenge_converted_from_issue'] = df_questions['Question_converted_from_issue']

challenge_content = df_questions['Question_title'] + \
    '. ' + df_questions['Question_body'].astype(str)
df_questions['Challenge_word_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_words(nlp(x)))
df_questions['Challenge_unique_word_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_unique_words(nlp(x)))
df_questions['Challenge_sentence_count'] = challenge_content.apply(
    lambda x: text_stats.basics.n_sents(nlp(x)))
df_questions['Challenge_information_entropy'] = challenge_content.apply(
    lambda x: text_stats.basics.entropy(nlp(x)))
df_questions['Challenge_readability'] = challenge_content.apply(
    lambda x: text_stats.readability.automated_readability_index(nlp(x)))
df_questions['Challenge_link_count'] = challenge_content.apply(
    lambda x: len(re.findall(link_pattern, x)))

df_questions['Solution_view_count'] = df_questions['Answer_comment_count']
df_questions['Solution_last_edit_time'] = df_questions['Answer_last_edit_time']
df_questions['Solution_score'] = df_questions['Answer_score']
df_questions['Solution_original_content'] = df_questions['Answer_original_content']
df_questions['Solution_preprocessed_content'] = df_questions['Answer_preprocessed_content']
df_questions['Solution_gpt_summary'] = df_questions['Answer_gpt_summary']

solution_content = df_questions['Answer_list'].apply(lambda x: '. '.join(x))
df_questions['Solution_word_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_words(nlp(x)))
df_questions['Solution_unique_word_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_unique_words(nlp(x)))
df_questions['Solution_sentence_count'] = solution_content.apply(
    lambda x: text_stats.basics.n_sents(nlp(x)))
df_questions['Solution_information_entropy'] = solution_content.apply(
    lambda x: text_stats.basics.entropy(nlp(x)))
df_questions['Solution_readability'] = solution_content.apply(
    lambda x: text_stats.readability.automated_readability_index(nlp(x)) if x else pd.NA)
df_questions['Solution_link_count'] = solution_content.apply(
    lambda x: len(re.findall(link_pattern, x)))

del df_issues['Issue_title']
del df_issues['Issue_body']
del df_issues['Issue_link']
del df_issues['Issue_creation_time']
del df_issues['Issue_answer_count']
del df_issues['Issue_upvote_count']
del df_issues['Issue_downvote_count']
del df_issues['Issue_original_content']
del df_issues['Issue_preprocessed_content']
del df_issues['Issue_gpt_summary_original']
del df_issues['Issue_gpt_summary']
del df_issues['Issue_closed_time']

del df_issues['Answer_list']
del df_issues['Answer_original_content']
del df_issues['Answer_preprocessed_content']
del df_issues['Answer_gpt_summary_original']
del df_issues['Answer_gpt_summary']

del df_questions['Question_title']
del df_questions['Question_body']
del df_questions['Question_link']
del df_questions['Question_creation_time']
del df_questions['Question_answer_count']
del df_questions['Question_comment_count']
del df_questions['Question_score']
del df_questions['Question_original_content']
del df_questions['Question_preprocessed_content']
del df_questions['Question_gpt_summary_original']
del df_questions['Question_gpt_summary']
del df_questions['Question_closed_time']
del df_questions['Question_view_count']
del df_questions['Question_favorite_count']
del df_questions['Question_last_edit_time']
del df_questions['Question_view_count']
del df_questions['Question_follower_count']
del df_questions['Question_converted_from_issue']

del df_questions['Answer_body']
del df_questions['Answer_list']
del df_questions['Answer_comment_count']
del df_questions['Answer_last_edit_time']
del df_questions['Answer_score']
del df_questions['Answer_original_content']
del df_questions['Answer_preprocessed_content']
del df_questions['Answer_gpt_summary_original']
del df_questions['Answer_gpt_summary']

df_all = pd.concat([df_issues, df_questions], ignore_index=True)
df_all.to_json(os.path.join(path_dataset, 'original.json'),
               indent=4, orient='records')

# remove custom stop words from challenges and solutions

from gensim.parsing.preprocessing import remove_stopwords

stop_words_custom = [
    'altern',
    'amazon',
    'answer',
    # 'api',
    'applic',
    'appreci',
    'approach',
    'aris',
    'ask',
    'assum',
    'attempt',
    'aw',
    'azur',
    'bad',
    # 'begin',
    'behavior',
    'behaviour',
    'best',
    'better',
    'case',
    'caus',
    'challeng',
    'cloudera',
    # 'close',
    'code',
    'command',
    'consid',
    'contain',
    'content',
    'correct',
    'correctli',
    'correspond',
    'couldn',
    'curiou',
    'custom',
    'deep',
    'demand',
    'demo',
    'despit',
    'differ',
    'differenti',
    'difficult',
    'difficulti',
    'discuss',
    'distinguish',
    'easi',
    'effect',
    'encount',
    # 'end',
    'enquiri',
    'error',
    'especi',
    'exampl',
    'expect',
    'experi',
    'databrick',
    'domo',
    'face',
    'fail',
    'failur',
    'favorit',
    'favourit',
    'feel',
    'firstli',
    'fix',
    'gcp',
    'given',
    'good',
    'googl',
    'gurante',
    'happen',
    'hard',
    'hei',
    'hello',
    'help',
    'ibm',
    'impli',
    'implic',
    'includ',
    'incorrect',
    'incorrectli',
    'indic',
    'info',
    'inform',
    'inquiri',
    'insight',
    'instead',
    'intern',
    'invalid',
    'issu',
    'lead',
    'learn',
    'like',
    'look',
    'machin',
    'main',
    'major',
    'manner',
    'mean',
    'meaning',
    'meaningfulli',
    'meaningless',
    'mention',
    'method',
    'microsoft',
    'mind',
    'mistak',
    'mistakenli',
    # 'multipl',
    'need',
    'new',
    'non',
    'occas',
    'occasion',
    'occur',
    'offer',
    'old',
    'own',
    # 'open',
    'oracl',
    'ought',
    'outcom',
    'particular',
    'particularli',
    'perspect',
    'point',
    'pointless',
    'possibl',
    'problem',
    'product',
    # 'program',
    'project',
    'provid',
    'python',
    'pytorch',
    'question',
    'refer',
    'regard',
    'requir',
    'resolv',
    'respond',
    'result',
    'right',
    'rightli',
    'scenario',
    'scikit',
    'script',
    'second',
    'secondli',
    'seek',
    'seen',
    'shall',
    'shan',
    'shouldn',
    'similar',
    'situat',
    'sklearn',
    'snippet',
    'snowflak',
    'solut',
    'solv',
    'sound',
    # 'sourc',
    'special',
    'specif',
    # 'start',
    'strang',
    'struggl',
    'succe',
    'success',
    'suggest',
    'talk',
    'tensorflow',
    'thank',
    'think',
    'thirdli',
    'thought',
    'topic',
    'try',
    'unabl',
    'understand',
    'unexpect',
    'us',
    'user',
    'usual',
    'valid',
    'view',
    'viewpoint',
    'wai',
    'want',
    'weird',
    'worst',
    'won',
    'wonder',
    'work',
    'wors',
    'wouldn',
    'wrong',
    'wrongli',
] 

df_all = pd.read_json(os.path.join(path_dataset, 'original.json'))

for index, row in df_all.iterrows():
    df_all.at[index, 'Challenge_original_content'] = remove_stopwords(row['Challenge_original_content'], stopwords=stop_words_custom)
    df_all.at[index, 'Challenge_preprocessed_content'] = remove_stopwords(row['Challenge_preprocessed_content'], stopwords=stop_words_custom)
    df_all.at[index, 'Challenge_summary'] = remove_stopwords(row['Challenge_summary'], stopwords=stop_words_custom)

    if row['Solution_gpt_summary']:
        df_all.at[index, 'Solution_original_content'] = remove_stopwords(row['Solution_original_content'], stopwords=stop_words_custom)
        df_all.at[index, 'Solution_preprocessed_content'] = remove_stopwords(row['Solution_preprocessed_content'], stopwords=stop_words_custom)
        df_all.at[index, 'Solution_gpt_summary'] = remove_stopwords(row['Solution_gpt_summary'], stopwords=stop_words_custom)

df_all.to_json(os.path.join(path_dataset, 'preprocessed.json'),
               indent=4, orient='records')

# remove issues with uninformed content

df_all = pd.read_json(os.path.join(path_dataset, 'preprocessed.json'))

for index, row in df_all.iterrows():
    if len(row['Challenge_original_content'].split()) < 6 or len(row['Challenge_original_content']) < 30:
        print(row['Challenge_original_content'])
        df_all.drop(index, inplace=True)
    elif row['Solution_original_content'] and (len(row['Solution_original_content'].split()) < 6 or len(row['Solution_original_content']) < 30):
        print(row['Solution_original_content'])
        df_all.drop(index, inplace=True)

df_all.to_json(os.path.join(path_dataset, 'filtered.json'),
               indent=4, orient='records')

# Draw sankey diagram of tool and platform

df_all = pd.read_json(os.path.join(path_dataset, 'filtered.json'))
df_all['State'] = df_all['Challenge_closed_time'].apply(lambda x: 'closed' if not pd.isna(x) else 'open')

categories = ['Tool', 'Platform', 'State']

df_all = df_all.groupby(categories).size().reset_index(name='value')
df_all.to_json(os.path.join(path_general, 'Tool platform info.json'),
               indent=4, orient='records')

newDf = pd.DataFrame()
for i in range(len(categories)-1):
    tempDf = df_all[[categories[i], categories[i+1], 'value']]
    tempDf.columns = ['source', 'target', 'value']
    newDf = pd.concat([newDf, tempDf])
newDf = newDf.groupby(['source', 'target']).agg({'value': 'sum'}).reset_index()

label = list(np.unique(df_all[categories].values))
source = newDf['source'].apply(lambda x: label.index(x))
target = newDf['target'].apply(lambda x: label.index(x))
value = newDf['value']

link = dict(source=source, target=target, value=value)
node = dict(label=label)
data = go.Sankey(link=link, node=node)

fig = go.Figure(data)
fig.update_layout(width=1000, height=1000, font_size=20)
fig.write_image(os.path.join(
    path_general, 'Tool platform sankey.png'))
