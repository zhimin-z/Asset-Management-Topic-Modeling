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

import numpy as np
import pandas as pd

df_issues = pd.read_json(os.path.join(path_dataset, 'issues.json'))

for index, row in df_issues.iterrows():
    df_issues.at[index, 'Challenge_title'] = row['Issue_title']
    df_issues.at[index, 'Challenge_body'] = row['Issue_body']
    df_issues.at[index, 'Challenge_link'] = row['Issue_link']
    df_issues.at[index, 'Challenge_original_content'] = row['Issue_original_content']
    df_issues.at[index, 'Challenge_preprocessed_content'] = row['Issue_preprocessed_content']
    df_issues.at[index, 'Challenge_gpt_summary'] = row['Issue_gpt_summary']
    df_issues.at[index, 'Challenge_created_time'] = row['Issue_created_time']
    df_issues.at[index, 'Challenge_answer_count'] = row['Issue_answer_count']
    df_issues.at[index, 'Challenge_score'] = row['Issue_upvote_count'] - row['Issue_downvote_count']
    df_issues.at[index, 'Challenge_closed_time'] = row['Issue_closed_time']

    df_issues.at[index, 'Solution_body'] = row['Answer_body']
    df_issues.at[index, 'Solution_score'] = row['Answer_score']
    df_issues.at[index, 'Solution_original_content'] = row['Answer_original_content']
    df_issues.at[index, 'Solution_preprocessed_content'] = row['Answer_preprocessed_content']
    df_issues.at[index, 'Solution_gpt_summary'] = row['Answer_gpt_summary']

del df_issues['Issue_title']
del df_issues['Issue_body']
del df_issues['Issue_link']
del df_issues['Issue_created_time']
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
del df_issues['Answer_score']
del df_issues['Answer_original_content']
del df_issues['Answer_preprocessed_content']
del df_issues['Answer_gpt_summary_original']
del df_issues['Answer_gpt_summary']

df_questions = pd.read_json(os.path.join(path_dataset, 'questions.json'))

for index, row in df_questions.iterrows():
    df_questions.at[index, 'Challenge_title'] = ['Question_title']
    df_questions.at[index, 'Challenge_body'] = ['Question_body']
    df_questions.at[index, 'Challenge_link'] = row['Question_link']
    df_questions.at[index, 'Challenge_original_content'] = row['Question_original_content']
    df_questions.at[index, 'Challenge_preprocessed_content'] = row['Question_preprocessed_content']
    df_questions.at[index, 'Challenge_gpt_summary'] = row['Question_gpt_summary']
    df_questions.at[index, 'Challenge_created_time'] = row['Question_created_time']
    df_questions.at[index, 'Challenge_answer_count'] = row['Question_answer_count']
    df_questions.at[index, 'Challenge_comment_count'] = row['Question_comment_count']
    df_questions.at[index, 'Challenge_score'] = row['Question_score']
    df_questions.at[index, 'Challenge_closed_time'] = row['Question_closed_time']
    df_questions.at[index, 'Challenge_favorite_count'] = row['Question_favorite_count']
    df_questions.at[index, 'Challenge_last_edit_time'] = row['Question_last_edit_time']
    df_questions.at[index, 'Challenge_view_count'] = row['Question_view_count']
    
    df_questions.at[index, 'Solution_body'] = row['Answer_body']
    df_questions.at[index, 'Solution_comment_count'] = row['Answer_comment_count']
    df_questions.at[index, 'Solution_last_edit_time'] = row['Answer_last_edit_time']
    df_questions.at[index, 'Solution_score'] = row['Answer_score']
    df_questions.at[index, 'Solution_original_content'] = row['Answer_original_content']
    df_questions.at[index, 'Solution_preprocessed_content'] = row['Answer_preprocessed_content']
    df_questions.at[index, 'Solution_gpt_summary'] = row['Answer_gpt_summary']

del df_questions['Question_title']
del df_questions['Question_body']
del df_questions['Question_link']
del df_questions['Question_created_time']
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

del df_questions['Answer_body']
del df_questions['Answer_list']
del df_questions['Answer_comment_count']
del df_questions['Answer_last_edit_time']
del df_questions['Answer_score']
del df_questions['Answer_original_content']
del df_questions['Answer_preprocessed_content']
del df_questions['Answer_gpt_summary_original']
del df_questions['Answer_gpt_summary']

df = pd.concat([df_issues, df_questions], ignore_index=True)
df.to_json(os.path.join(path_dataset, 'original.json'),
               indent=4, orient='records')

# Draw sankey diagram of tool and platform

import plotly.graph_objects as go

df = pd.read_json(os.path.join(path_dataset, 'original.json'))
df['State'] = df['Challenge_closed_time'].apply(lambda x: 'closed' if not pd.isna(x) else 'open')

categories = ['Platform', 'Tool', 'State']

df_info = df.groupby(categories).size().reset_index(name='value')
df_info.to_json(os.path.join(path_general, 'Tool platform state info.json'),
               indent=4, orient='records')

labels = {}
newDf = pd.DataFrame()
for i in range(len(categories)):
    labels.update(df[categories[i]].value_counts().to_dict())
    if i == len(categories)-1:
        break
    tempDf = df_info[[categories[i], categories[i+1], 'value']]
    tempDf.columns = ['source', 'target', 'value']
    newDf = pd.concat([newDf, tempDf])
    
newDf = newDf.groupby(['source', 'target']).agg({'value': 'sum'}).reset_index()
source = newDf['source'].apply(lambda x: list(labels).index(x))
target = newDf['target'].apply(lambda x: list(labels).index(x))
value = newDf['value']

labels = [f'{k} ({v})' for k, v in labels.items()]

link = dict(source=source, target=target, value=value)
node = dict(label=labels)
data = go.Sankey(link=link, node=node)

fig = go.Figure(data)
fig.update_layout(width=1000, height=1000, font_size=20)
fig.write_image(os.path.join(
    path_general, 'Tool platform state sankey.png'))

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

df = pd.read_json(os.path.join(path_dataset, 'original.json'))

for index, row in df.iterrows():
    Challenge_original_content = remove_stopwords(row['Challenge_original_content'], stopwords=stop_words_custom)
    if len(Challenge_original_content.split()) > 4:
        df.at[index, 'Challenge_original_content'] = Challenge_original_content
    else:
        df.at[index, 'Challenge_original_content'] = np.nan
        
    Challenge_preprocessed_content = remove_stopwords(row['Challenge_preprocessed_content'], stopwords=stop_words_custom)
    if len(Challenge_preprocessed_content.split()) > 4:
        df.at[index, 'Challenge_preprocessed_content'] = Challenge_preprocessed_content
    else:
        df.at[index, 'Challenge_preprocessed_content'] = np.nan
        
    Challenge_gpt_summary = remove_stopwords(row['Challenge_gpt_summary'], stopwords=stop_words_custom)
    if len(Challenge_gpt_summary.split()) > 4:
        df.at[index, 'Challenge_gpt_summary'] = Challenge_gpt_summary
    else:
        df.at[index, 'Challenge_gpt_summary'] = np.nan

    if pd.notna(row['Solution_gpt_summary']):
        Solution_original_content = remove_stopwords(row['Solution_original_content'], stopwords=stop_words_custom)
        if len(Solution_original_content.split()) > 4:
            df.at[index, 'Solution_original_content'] = Solution_original_content
        else:
            df.at[index, 'Solution_original_content'] = np.nan

        Solution_preprocessed_content = remove_stopwords(row['Solution_preprocessed_content'], stopwords=stop_words_custom)
        if len(Solution_preprocessed_content.split()) > 4:
            df.at[index, 'Solution_preprocessed_content'] = Solution_preprocessed_content
        else:
            df.at[index, 'Solution_preprocessed_content'] = np.nan
            
        Solution_gpt_summary = remove_stopwords(row['Solution_gpt_summary'], stopwords=stop_words_custom)
        if len(Solution_gpt_summary.split()) > 4:
            df.at[index, 'Solution_gpt_summary'] = Solution_gpt_summary
        else:
            df.at[index, 'Solution_gpt_summary'] = np.nan

df.to_json(os.path.join(path_dataset, 'preprocessed.json'),
               indent=4, orient='records')
