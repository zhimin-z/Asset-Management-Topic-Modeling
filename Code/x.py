# Compare metrics evolution of Closed vs Open challenges across different topics

df_challenge = pd.read_json(os.path.join(path_general, 'filtered.json'))

fig_challenge_topic_count_closed = go.Figure()
fig_challenge_view_count_closed = go.Figure()
fig_challenge_favorite_count_closed = go.Figure()
fig_challenge_answer_count_closed = go.Figure()
fig_challenge_comment_count_closed = go.Figure()
fig_challenge_participation_count_closed = go.Figure()

fig_challenge_topic_count_open = go.Figure()
fig_challenge_view_count_open = go.Figure()
fig_challenge_favorite_count_open = go.Figure()
fig_challenge_answer_count_open = go.Figure()
fig_challenge_comment_count_open = go.Figure()
fig_challenge_participation_count_open = go.Figure()

for name, group in df_challenge.groupby('Challenge_topic_macro'):
    closed = group[group['Tool'] == 'Amazon Closed']
    open = group[group['Tool'] == 'Azure Machine Learning']

    # plot challenge topic count over time
    group_closed = closed.groupby(pd.Grouper(key='Challenge_created_time', freq='Y'))[
        'Challenge_topic_macro'].count().reset_index()
    x_closed = pd.to_datetime(
        group_closed['Challenge_created_time']).values
    y_closed = group_closed['Challenge_topic_macro'].values
    diff_y = np.diff(y_closed)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_topic_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    group_open = open.groupby(pd.Grouper(key='Challenge_created_time', freq='Y'))[
        'Challenge_topic_macro'].count().reset_index()
    x_open = pd.to_datetime(group_open['Challenge_created_time']).values
    y_open = group_open['Challenge_topic_macro'].values
    diff_y = np.diff(y_open)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_topic_count_open.add_trace(
        go.Scatter(x=x_open, y=diff_y, mode='lines', name=name))

    # plot challenge participation count over time
    group_closed = closed.groupby(pd.Grouper(key='Challenge_created_time', freq='Y'))[['Challenge_participation_count', 'Challenge_answer_count', 'Challenge_comment_count', 'Challenge_view_count', 'Challenge_favorite_count', 'Challenge_link_count', 'Challenge_word_count', 'Challenge_score', 'Challenge_unique_word_count', 'Challenge_sentence_count', 'Challenge_information_entropy', 'Challenge_readability']].sum().reset_index()
    y = group_closed['Challenge_participation_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_participation_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    group_open = open.groupby(pd.Grouper(key='Challenge_created_time', freq='Y'))[['Challenge_participation_count', 'Challenge_answer_count', 'Challenge_comment_count', 'Challenge_view_count', 'Challenge_favorite_count', 'Challenge_link_count', 'Challenge_word_count', 'Challenge_score', 'Challenge_unique_word_count', 'Challenge_sentence_count', 'Challenge_information_entropy', 'Challenge_readability']].sum().reset_index()
    y = group_open['Challenge_participation_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_participation_count_open.add_trace(
        go.Scatter(x=x_open, y=diff_y, mode='lines', name=name))

    # plot challenge answer count over time
    y = group_closed['Challenge_answer_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_answer_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))
    
    # plot challenge comment count over time
    y = group_closed['Challenge_comment_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_comment_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    # plot challenge view count over time
    y = group_closed['Challenge_view_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_view_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    y = group_open['Challenge_view_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_view_count_open.add_trace(
        go.Scatter(x=x_open, y=diff_y, mode='lines', name=name))

    # plot challenge favorite count over time
    y = group_closed['Challenge_favorite_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_favorite_count_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    y = group_open['Challenge_favorite_count'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_favorite_count_open.add_trace(
        go.Scatter(x=x_open, y=diff_y, mode='lines', name=name))

    # plot challenge score over time
    y = group_closed['Challenge_score'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_score_closed.add_trace(
        go.Scatter(x=x_closed, y=diff_y, mode='lines', name=name))

    y = group_open['Challenge_score'].values
    diff_y = np.diff(y)
    diff_y = np.insert(diff_y, 0, 0)
    fig_challenge_score_open.add_trace(
        go.Scatter(x=x_open, y=diff_y, mode='lines', name=name))

    # plot challenge link count over time
    y = group_closed['Challenge_link_count'].values / y_closed
    fig_challenge_link_count_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_link_count'].values / y_open
    fig_challenge_link_count_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

    # plot challenge word count over time
    y = group_closed['Challenge_word_count'].values / y_closed
    fig_challenge_word_count_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_word_count'].values / y_open
    fig_challenge_word_count_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

    # plot challenge sentence count over time
    y = group_closed['Challenge_sentence_count'].values / y_closed
    fig_challenge_sentence_count_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_sentence_count'].values / y_open
    fig_challenge_sentence_count_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

    # plot challenge unique word count over time
    y = group_closed['Challenge_unique_word_count'].values / y_closed
    fig_challenge_unique_word_count_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_unique_word_count'].values / y_open
    fig_challenge_unique_word_count_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

    # plot challenge information entropy over time
    y = group_closed['Challenge_information_entropy'].values / y_closed
    fig_challenge_information_entropy_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_information_entropy'].values / y_open
    fig_challenge_information_entropy_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

    # plot challenge readability over time
    y = group_closed['Challenge_readability'].values / y_closed
    fig_challenge_readability_closed.add_trace(
        go.Scatter(x=x_closed, y=y, mode='lines', name=name))

    y = group_open['Challenge_readability'].values / y_open
    fig_challenge_readability_open.add_trace(
        go.Scatter(x=x_open, y=y, mode='lines', name=name))

fig_challenge_topic_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_view_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_favorite_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_participation_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_answer_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_comment_count_closed.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))

fig_challenge_topic_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_view_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_favorite_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_answer_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_comment_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))
fig_challenge_participation_count_open.update_layout(
    width=2000,
    height=1000,
    margin=dict(l=0, r=0, t=0, b=0))

fig_challenge_topic_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_topic_count_increase_rate (Closed).png'))
fig_challenge_view_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_view_count_increase_rate (Closed).png'))
fig_challenge_favorite_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_favorite_count_increase_rate (Closed).png'))
fig_challenge_answer_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_answer_count_increase_rate (Closed).png'))
fig_challenge_comment_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_comment_count_increase_rate (Closed).png'))
fig_challenge_participation_count_closed.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_participation_count_increase_rate (Closed).png'))

fig_challenge_topic_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_topic_count_increase_rate (Open).png'))
fig_challenge_view_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_view_count_increase_rate (Open).png'))
fig_challenge_favorite_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_favorite_count_increase_rate (Open).png'))
fig_challenge_answer_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_answer_count_increase_rate (Open).png'))
fig_challenge_comment_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_comment_count_increase_rate (Open).png'))
fig_challenge_participation_count_open.write_image(os.path.join(
    path_challenge_open_closed, f'Challenge_participation_count_increase_rate (Open).png'))