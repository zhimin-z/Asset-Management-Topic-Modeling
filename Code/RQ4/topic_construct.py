import os
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

from plotly.subplots import make_subplots

path_rq4 = os.path.join(os.getcwd(), 'Result', 'RQ4')
df = pd.read_json(os.path.join(path_rq4, 'macro-topics.json'))

macro_topic_indexing = {
    0: 'Code Development',
    1: 'Code Management',
    2: 'Comparison & Recommendation',
    3: 'Compute Management',
    4: 'Data Development',
    5: 'Data Management',
    6: 'Environment Management',
    7: 'Experiment Management',
    8: 'File Management',
    9: 'Maintenance & Support',
    10: 'Model Development',
    11: 'Model Management',
    12: 'Model Deployment',
    13: 'Network Management',
    14: 'Observability Management',
    15: 'Pipeline Management',
    16: 'Security Management',
    17: 'User Interface Management'
}
color_map = {
    'Problem': 'tomato',
    'Knowledge': 'dodgerblue',
}
rows = 5
cols = 4
fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05, subplot_titles=[
                    macro_topic_indexing[i] for i in sorted(df['Resolution_summary_topic_macro'].unique())])

for macro_name, macro_group in df.groupby('Resolution_summary_topic_macro', sort=True):
    categories = []
    frequency_p = []
    frequency_k = []

    for name, group in macro_group.groupby('Resolution_summary_topic'):
        name = r'$\hat{R}_{0' + str(name+1) + '}$' if name < 9 else r'$\hat{R}_{' + str(name+1) + '}$'
        categories.append(name)
        frequency_p.append(len(group[group['Challenge_type'] == 'problem'])/len(group)*100)
        frequency_k.append(len(group[group['Challenge_type'] == 'knowledge'])/len(group)*100)

    row = macro_name // cols + 1
    col = macro_name % cols + 1
    show_legend = True if macro_name == 0 else False

    fig.add_trace(go.Bar(
        name='Problem',
        x=categories,
        y=frequency_p,
        legendgroup='Problem',
        marker_color=color_map['Problem'],
        showlegend=show_legend
    ), row=row, col=col)
    fig.add_trace(go.Bar(
        name='Knowledge',
        x=categories,
        y=frequency_k,
        legendgroup='Knowledge',
        marker_color=color_map['Knowledge'],
        showlegend=show_legend
    ), row=row, col=col)
    fig.update_xaxes(
        tickangle=90,
        tickfont=dict(size=10),
        row=row,
        col=col
    )

fig.update_yaxes(range=[0, 100])
fig.update_layout(
    barmode='group',
    width=1200,
    height=1000,
    margin=go.layout.Margin(
        l=20,  # left margin
        r=20,  # right margin
        b=20,  # bottom margin
        t=20,  # top margin
    )
)
fig.update_annotations(dict(font_size=13))
pio.full_figure_for_development(fig, warn=False)
fig.show()
fig.write_image(os.path.join(path_rq4, 'Macro-topics group frequency histogram.pdf'))