import os
import math
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

path_rq12 = os.path.join(os.getcwd(), 'Result', 'RQ12')
df = pd.read_json(os.path.join(path_rq12, 'macro-topics.json'))

macro_topic_indexing = {
    0: 'Code Development',
    1: 'Code Management',
    2: 'Compute Management',
    3: 'Data Development',
    4: 'Data Management',
    5: 'Environment Management',
    6: 'Experiment Management',
    7: 'File Management',
    8: 'Model Development',
    9: 'Model Management',
    10: 'Model Deployment',
    11: 'Network Management',
    12: 'Observability Management',
    13: 'Pipeline Management',
    14: 'Security Management',
    15: 'User Interface Management',
}

color_map = {
    'Problem': 'tomato',
    'Knowledge': 'dodgerblue',
}
rows = cols = math.ceil(math.sqrt(df['Challenge_topic_macro'].nunique()))
fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.05, vertical_spacing=0.05, subplot_titles=[macro_topic_indexing[i] for i in sorted(df['Challenge_topic_macro'].unique())])

for macro_name, macro_group in df.groupby('Challenge_topic_macro', sort=True):
    categories = []
    frequency_p = []
    frequency_k = [] 
    
    for name, group in macro_group.groupby('Challenge_topic'):
        name = r'$\hat{C}_{0' + str(name+1) + '}$' if name < 9 else r'$\hat{C}_{' + str(name+1) + '}$'
        categories.append(f'C{name}')
        frequency_p.append(len(group[group['Challenge_type'] == 'problem'])/len(group)*100)
        frequency_k.append(len(group[group['Challenge_type'] == 'knowledge'])/len(group)*100)
    
    row = macro_name // rows + 1
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
    barmode='stack',
    width=1000, 
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
fig.write_image(os.path.join(path_rq12, 'Macro-topics group frequency histogram.pdf'))
