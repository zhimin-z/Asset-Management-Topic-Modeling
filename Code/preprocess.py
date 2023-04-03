import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
import numpy as np
import pandas as pd

import subprocess
cmd_str = "python -m spacy download en_core_web_trf -q"
subprocess.run(cmd_str, shell=True)

# Refer to https://textacy.readthedocs.io/en/stable/api_reference/text_stats.html
from textacy import text_stats

nlp = spacy.load('en_core_web_trf')
link_pattern = '(?P<url>ftp|https?://[^\s]+)'

df_issues = pd.read_json(os.path.join(path_dataset, 'issues.json'))

df_issues['Solution_word_count'] = np.nan
df_issues['Solution_unique_word_count'] = np.nan
df_issues['Solution_sentence_count'] = np.nan
df_issues['Solution_information_entropy'] = np.nan
df_issues['Solution_readability'] = np.nan
df_issues['Solution_link_count'] = np.nan

for index, row in df_issues.iterrows():
    df_issues.at[index, 'Challenge_link'] = row['Issue_link']
    df_issues.at[index, 'Challenge_original_content'] = row['Issue_original_content']
    df_issues.at[index, 'Challenge_preprocessed_content'] = row['Issue_preprocessed_content']
    df_issues.at[index, 'Challenge_gpt_summary'] = row['Issue_gpt_summary']
    df_issues.at[index, 'Challenge_creation_time'] = row['Issue_creation_time']
    df_issues.at[index, 'Challenge_answer_count'] = row['Issue_answer_count']
    df_issues.at[index, 'Challenge_score'] = row['Issue_upvote_count'] - row['Issue_downvote_count']
    df_issues.at[index, 'Challenge_closed_time'] = row['Issue_closed_time']
    
    challenge_content = row['Issue_title'] + '. ' + str(row['Issue_body'])
    challenge_content_nlp = nlp(challenge_content)
    df_issues.at[index, 'Challenge_word_count'], df_issues.at[index, 'Challenge_unique_word_count'] = text_stats.utils.compute_n_words_and_types(challenge_content_nlp)
    df_issues.at[index, 'Challenge_sentence_count'] = text_stats.basics.n_sents(challenge_content_nlp)
    df_issues.at[index, 'Challenge_information_entropy'] = text_stats.basics.entropy(challenge_content_nlp)
    df_issues.at[index, 'Challenge_readability'] = text_stats.readability.automated_readability_index(challenge_content_nlp)
    df_issues.at[index, 'Challenge_link_count'] = len(re.findall(link_pattern, challenge_content))
    
    df_issues.at[index, 'Solution_original_content'] = row['Answer_original_content']
    df_issues.at[index, 'Solution_preprocessed_content'] = row['Answer_preprocessed_content']
    df_issues.at[index, 'Solution_gpt_summary'] = row['Answer_gpt_summary']
    
    discussion = row['Answer_body']
    
    if pd.notna(discussion):
        discussion_nlp = nlp(discussion)
        df_issues.at[index, 'Solution_word_count'], df_issues.at[index, 'Solution_unique_word_count'] = text_stats.utils.compute_n_words_and_types(discussion_nlp)
        df_issues.at[index, 'Solution_sentence_count'] = text_stats.basics.n_sents(discussion_nlp)
        df_issues.at[index, 'Solution_information_entropy'] = text_stats.basics.entropy(discussion_nlp)
        df_issues.at[index, 'Solution_readability'] = text_stats.readability.automated_readability_index(discussion_nlp)
        df_issues.at[index, 'Solution_link_count'] = len(re.findall(link_pattern, discussion))

df_questions = pd.read_json(os.path.join(path_dataset, 'questions.json'))

df_questions['Solution_word_count'] = np.nan
df_questions['Solution_unique_word_count'] = np.nan
df_questions['Solution_sentence_count'] = np.nan
df_questions['Solution_information_entropy'] = np.nan
df_questions['Solution_readability'] = np.nan
df_questions['Solution_link_count'] = np.nan

for index, row in df_questions.iterrows():
    df_questions.at[index, 'Challenge_link'] = row['Question_link']
    df_questions.at[index, 'Challenge_original_content'] = row['Question_original_content']
    df_questions.at[index, 'Challenge_preprocessed_content'] = row['Question_preprocessed_content']
    df_questions.at[index, 'Challenge_gpt_summary'] = row['Question_gpt_summary']
    df_questions.at[index, 'Challenge_creation_time'] = row['Question_creation_time']
    df_questions.at[index, 'Challenge_answer_count'] = row['Question_answer_count']
    df_questions.at[index, 'Challenge_comment_count'] = row['Question_comment_count']
    df_questions.at[index, 'Challenge_score'] = row['Question_score']
    df_questions.at[index, 'Challenge_closed_time'] = row['Question_closed_time']
    df_questions.at[index, 'Challenge_favorite_count'] = row['Question_favorite_count']
    df_questions.at[index, 'Challenge_last_edit_time'] = row['Question_last_edit_time']
    df_questions.at[index, 'Challenge_view_count'] = row['Question_view_count']
    df_questions.at[index, 'Challenge_follower_count'] = row['Question_follower_count']
    df_questions.at[index, 'Challenge_converted_from_issue'] = row['Question_converted_from_issue']
    
    challenge_content = row['Question_title'] + '. ' + str(row['Question_body'])
    challenge_content_nlp = nlp(challenge_content)
    df_questions.at[index, 'Challenge_word_count'], df_questions.at[index, 'Challenge_unique_word_count'] = text_stats.utils.compute_n_words_and_types(challenge_content_nlp)
    df_questions.at[index, 'Challenge_sentence_count'] = text_stats.basics.n_sents(challenge_content_nlp)
    df_questions.at[index, 'Challenge_information_entropy'] = text_stats.basics.entropy(challenge_content_nlp)
    df_questions.at[index, 'Challenge_readability'] = text_stats.readability.automated_readability_index(challenge_content_nlp)
    df_questions.at[index, 'Challenge_link_count'] = len(re.findall(link_pattern, challenge_content))
    
    df_questions.at[index, 'Solution_comment_count'] = row['Answer_comment_count']
    df_questions.at[index, 'Solution_last_edit_time'] = row['Answer_last_edit_time']
    df_questions.at[index, 'Solution_score'] = row['Answer_score']
    df_questions.at[index, 'Solution_original_content'] = row['Answer_original_content']
    df_questions.at[index, 'Solution_preprocessed_content'] = row['Answer_preprocessed_content']
    df_questions.at[index, 'Solution_gpt_summary'] = row['Answer_gpt_summary']
    
    discussion = row['Answer_body']
        
    if discussion:
        discussion_nlp = nlp(discussion)
        df_questions.at[index, 'Solution_word_count'], df_questions.at[index, 'Solution_unique_word_count'] = text_stats.utils.compute_n_words_and_types(discussion_nlp)
        df_questions.at[index, 'Solution_sentence_count'] = text_stats.basics.n_sents(discussion_nlp)
        df_questions.at[index, 'Solution_information_entropy'] = text_stats.basics.entropy(discussion_nlp)
        df_questions.at[index, 'Solution_readability'] = text_stats.readability.automated_readability_index(discussion_nlp)
        df_questions.at[index, 'Solution_link_count'] = len(re.findall(link_pattern, discussion))

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

del df_issues['Answer_body']
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
df_all = df_all.reindex(sorted(df_all.columns), axis=1)
df_all.to_json(os.path.join(path_dataset, 'original.json'),
               indent=4, orient='records')

# Draw sankey diagram of tool and platform

import plotly.graph_objects as go

df_all = pd.read_json(os.path.join(path_dataset, 'original.json'))
df_all['State'] = df_all['Challenge_closed_time'].apply(lambda x: 'closed' if not pd.isna(x) else 'open')

categories = ['Platform', 'Tool', 'State']

df_all = df_all.groupby(categories).size().reset_index(name='value')
df_all.to_json(os.path.join(path_general, 'Tool platform state info.json'),
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
    path_general, 'Tool platform state sankey.png'))

df_all = pd.read_json(os.path.join(path_general, 'Tool platform info.json'))
df_all = df_all.groupby(['Platform', 'State']).agg({'value': 'sum'}).reset_index()
df_all.to_json(os.path.join(path_general, 'Platform state info.json'),
               indent=4, orient='records')

# remove custom stop words from challenges and solutions

from gensim.parsing.preprocessing import remove_stopwords

stop_words_custom = [
    'abl',
    'acknowledg',
    'actual',
    'addition',
    'admit',
    'advis',
    # 'allow',
    'alright',
    'altern',
    'amaz',
    'amazon',
    'answer',
    # 'api',
    'appear',
    'applic',
    'appreci',
    'approach',
    'appropri',
    # 'approv',
    'aris',
    'ask',
    'assum',
    'astonish',
    'attempt',
    'aw',
    'awesom',
    'azur',
    'bad',
    # 'begin',
    'behavior',
    'behaviour',
    'best',
    'better',
    'bring',
    'bug',
    'case',
    'categori',
    'caus',
    'certain',
    'challeng',
    'chang',
    'check',
    # 'close',
    'cloudera',
    'code',
    # 'colab',
    'command',
    'concern',
    'confirm',
    'confus',
    'consid',
    'contain',
    'content',
    'correct',
    'correctli',
    'correspond',
    'couldn',
    'curiou',
    'current',
    'custom',
    'deep',
    'demand',
    'demo',
    'depict',
    'describ',
    'despit',
    'detail',
    'develop',
    'differ',
    'differenti',
    'difficult',
    'difficulti',
    'discov',
    'discuss',
    'distinguish',
    'easi',
    'effect',
    'emerg',
    'encount',
    # 'end',
    'enquiri',
    'ensur',
    'error',
    'especi',
    'exampl',
    'excit',
    'expect',
    'experi',
    'eventu',
    'databrick',
    'domo',
    'face',
    'fact',
    'fascin',
    'fail',
    'failur',
    'fairli',
    'favorit',
    'favourit',
    'feel',
    'find',
    'firstli',
    'fix',
    'follow',
    'form',
    'gcp',
    'get',
    'given',
    'good',
    'googl',
    'guarante',
    'happen',
    'hard',
    'hear',
    'hei',
    'hello',
    'help',
    'ibm',
    'impli',
    'implic',
    'includ',
    'incorrect',
    'incorrectli',
    'incred',
    'indic',
    'info',
    'inform',
    'inquiri',
    'insight',
    'instead',
    'interest',
    'invalid',
    'investig',
    'issu',
    'join',
    # 'jupyter',
    # 'keras',
    'kind',
    'know',
    'known',
    'lead',
    'learn',
    'let',
    'like',
    'look',
    'machin',
    'make',
    'main',
    'major',
    'manag',
    'manner',
    'marvel',
    'mean',
    'meaning',
    'meaningfulli',
    'meaningless',
    'meantim',
    'mention',
    'method',
    'microsoft',
    'mind',
    'mistak',
    'mistakenli',
    # 'multipl',
    'near',
    'necessari',
    'need',
    'new',
    'non',
    'notice',
    'obtain',
    'occas',
    'occasion',
    'occur',
    'offer',
    'old',
    'opinion',
    'own',
    # 'open',
    'oracl',
    'ought',
    'outcom',
    'part',
    'particip',
    'particular',
    'particularli',
    'perceive',
    # 'perform',
    'permit',
    'person',
    'perspect',
    'point',
    'pointless',
    'possibl',
    'post',
    'pretty',
    'problem',
    'product',
    # 'program',
    'project',
    'proper',
    'provid',
    'python',
    # 'pytorch',
    'question',
    'realize',
    'recognize',
    'recommend',
    'refer',
    'regard',
    'requir',
    'resolv',
    'respond',
    'result',
    'right',
    'rightli',
    'saw',
    'scenario',
    # 'scikit',
    'script',
    'second',
    'secondli',
    'seek',
    'seen',
    'shall',
    'shan',
    'shock',
    'shouldn',
    'similar',
    'situat',
    # 'sklearn',
    'snippet',
    'snowflak',
    'solut',
    'solv',
    'sound',
    # 'sourc',
    'special',
    'specif',
    # 'start',
    'startl',
    'strang',
    'struggl',
    'stun',
    'succe',
    'success',
    'suggest',
    'super',
    'support',
    'sure',
    'suspect',
    'take',
    'talk',
    # 'tensorflow',
    'thank',
    'think',
    'thirdli',
    'thought',
    'tool',
    'topic',
    # 'transformers',
    'truth',
    'try',
    'unabl',
    'understand',
    'unexpect',
    'unsur',
    'upcom',
    'us',
    'user',
    'usual',
    'valid',
    'view',
    'viewpoint',
    'wai',
    'want',
    'weird',
    'will',
    'worst',
    'won',
    'wonder',
    'work',
    'wors',
    'wouldn',
    'wrong',
    'wrongli',
    'xgboost',
    'ye',
] 

df_all = pd.read_json(os.path.join(path_dataset, 'original.json'))

for index, row in df_all.iterrows():
    df_all.at[index, 'Challenge_original_content'] = remove_stopwords(row['Challenge_original_content'], stopwords=stop_words_custom)
    df_all.at[index, 'Challenge_preprocessed_content'] = remove_stopwords(row['Challenge_preprocessed_content'], stopwords=stop_words_custom)
    df_all.at[index, 'Challenge_gpt_summary'] = remove_stopwords(row['Challenge_gpt_summary'], stopwords=stop_words_custom)

    if row['Solution_gpt_summary']:
        df_all.at[index, 'Solution_original_content'] = remove_stopwords(row['Solution_original_content'], stopwords=stop_words_custom)
        df_all.at[index, 'Solution_preprocessed_content'] = remove_stopwords(row['Solution_preprocessed_content'], stopwords=stop_words_custom)
        df_all.at[index, 'Solution_gpt_summary'] = remove_stopwords(row['Solution_gpt_summary'], stopwords=stop_words_custom)

df_all.to_json(os.path.join(path_dataset, 'preprocessed.json'),
               indent=4, orient='records')
