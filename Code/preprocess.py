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
cmd_str = "python -m spacy download en_core_web_trf"
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

nlp = spacy.load('en_core_web_trf')
link_pattern = '(?P<url>ftp|https?://[^\s]+)'

df_issues = pd.read_json(os.path.join(path_dataset, 'issues.json'))

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
    
    discussion = '. '.join(row['Answer_list'])
    
    if discussion:
        discussion_nlp = nlp(discussion)
        df_issues.at[index, 'Solution_word_count'], df_issues.at[index, 'Solution_unique_word_count'] = text_stats.utils.compute_n_words_and_types(discussion_nlp)
        df_issues.at[index, 'Solution_sentence_count'] = text_stats.basics.n_sents(discussion_nlp)
        df_issues.at[index, 'Solution_information_entropy'] = text_stats.basics.entropy(discussion_nlp)
        df_issues.at[index, 'Solution_readability'] = text_stats.readability.automated_readability_index(discussion_nlp)
        df_issues.at[index, 'Solution_link_count'] = len(re.findall(link_pattern, discussion))
    else:
        df_issues.at[index, 'Solution_word_count'] = np.nan
        df_issues.at[index, 'Solution_unique_word_count'] = np.nan
        df_issues.at[index, 'Solution_sentence_count'] = np.nan
        df_issues.at[index, 'Solution_information_entropy'] = np.nan
        df_issues.at[index, 'Solution_readability'] = np.nan
        df_issues.at[index, 'Solution_link_count'] = np.nan

df_questions = pd.read_json(os.path.join(path_dataset, 'questions.json'))

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
    
    discussion = ''
    if row['Answer_body']:
        discussion = row['Answer_body']
    elif row['Answer_list']:
        if pd.notna(row['Question_closed_time']):
            for comment in row['Answer_list']:
                if comment['Answer_has_accepted']:
                    discussion = comment['Answer_body']
                    break
        else:
            discussion = '. '.join(row['Answer_list'])
        
    if discussion:
        discussion_nlp = nlp(discussion)
        df_questions.at[index, 'Solution_word_count'], df_questions.at[index, 'Solution_unique_word_count'] = text_stats.utils.compute_n_words_and_types(discussion_nlp)
        df_questions.at[index, 'Solution_sentence_count'] = text_stats.basics.n_sents(discussion_nlp)
        df_questions.at[index, 'Solution_information_entropy'] = text_stats.basics.entropy(discussion_nlp)
        df_questions.at[index, 'Solution_readability'] = text_stats.readability.automated_readability_index(discussion_nlp)
        df_questions.at[index, 'Solution_link_count'] = len(re.findall(link_pattern, discussion))
    else:
        df_questions.at[index, 'Solution_word_count'] = np.nan
        df_questions.at[index, 'Solution_unique_word_count'] = np.nan
        df_questions.at[index, 'Solution_sentence_count'] = np.nan
        df_questions.at[index, 'Solution_information_entropy'] = np.nan
        df_questions.at[index, 'Solution_readability'] = np.nan
        df_questions.at[index, 'Solution_link_count'] = np.nan

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