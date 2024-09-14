from pathlib import Path

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.io as pio


data_folder = Path('data')
df = pd.read_csv(data_folder / 'combined_results.csv')
regression_df = pd.read_csv(data_folder / 'regression_results.csv')
issue_df = pd.read_csv(data_folder / 'issue_frequency_results.csv')

pio.templates.default = "plotly_white"
color_sequence = px.colors.qualitative.D3

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label('', style={'font-weight': 'bold', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='product-selector',
            options=[{'label': p, 'value': p} for p in df['product_name'].unique()],
            value=df['product_name'].unique()[0],
            clearable=False,
            style={
                'width': '700px',
                'verticalAlign': 'middle',
                'border': '1px solid #ccc',
                'border-radius': '4px',
                'padding': '5px'
            }
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'margin-bottom': '20px'}),
    dcc.Graph(id='regression-bar-chart'),
    dcc.Graph(id='issue-bar-chart'),
    dcc.Graph(id='scatter-plot', config={'modeBarButtonsToAdd': ['lasso2d']}),
    dash_table.DataTable(
        id='data-table',
        page_size=10,
        style_table={'width': '100%'},
        style_cell={
            'textAlign': 'left',
            'whiteSpace': 'normal',
            'height': 'auto',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': '500px',
        },
        style_header={
            'backgroundColor': '#f2f2f2',
            'fontWeight': 'bold'
        },
        style_data={
            'border': '1px solid #ddd',
            'padding': '5px'
        }
    ),
])


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('product-selector', 'value')]
)
def update_scatter_plot(selected_product):
    filtered_df = (df
                   .query('product_name == @selected_product')
                   .rename(columns={'cluster_name': 'Issue Cluster'})
            )
    fig = px.scatter(filtered_df,
                     x='emb_1',
                     y='emb_2',
                     color='Issue Cluster',
                     hover_data={'issue': True},
                     color_discrete_sequence=color_sequence)
    fig.update_layout(dragmode='lasso',
                      xaxis_title=None,
                      yaxis_title=None,
                      xaxis_visible=False,
                      yaxis_visible=False)
    return fig


@app.callback(
    Output('data-table', 'data'),
    [Input('scatter-plot', 'selectedData'),
     Input('product-selector', 'value')]
)
def update_table(selectedData, selected_product):
    filtered_df = df.query('product_name == @selected_product')
    if selectedData:
        selected_points = [point['pointIndex'] for point in selectedData['points']]
        filtered_df = filtered_df.iloc[selected_points]

    filtered_df = (filtered_df
                   .query('product_name == @selected_product')
                   .filter(items=['review', 'rating', 'n_review_votes'])
                   .drop_duplicates(subset='review')
                   .assign(rating= lambda x: x.rating.apply(rating_to_stars))
                   .rename(columns={'review': 'Review',
                                    'rating': 'Rating',
                                    'n_review_votes': 'Number of Votes'})
                   )

    return filtered_df.to_dict('records')


@app.callback(
    Output('regression-bar-chart', 'figure'),
    [Input('product-selector', 'value')]
)
def update_regression_bar_chart(selected_product):
    top_5_reg = (regression_df
                   .query('product_name == @selected_product')
                   .assign(impact= lambda x: x['impact'].abs())
                   .query('issue != "Other"')
                   .nlargest(5, 'impact')
                )
    fig = px.bar(top_5_reg,
                 x='impact',
                 y='issue',
                 orientation='h',
                 color='issue',
                 color_discrete_sequence=color_sequence
                 )
    fig.update_layout(
        title='Potential Rating Improvement by fixing Issue',
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False
    )
    return fig


@app.callback(
    Output('issue-bar-chart', 'figure'),
    [Input('product-selector', 'value')]
)
def update_issue_bar_chart(selected_product):
    top_5_issues = (issue_df
                    .query('product_name == @selected_product')
                    .query('issue != "Other"')
                    .assign(frequency_difference= lambda x: x.frequency_difference * 100)
                    .nlargest(5, 'frequency_difference')
                )
    fig = px.bar(top_5_issues,
                 y='issue',
                 x='frequency_difference',
                 orientation='h',
                 color='issue',
                 color_discrete_sequence=color_sequence)
    fig.update_layout(
        title='Difference in Issue Frequency between Product and Average',
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False
    )
    fig.update_xaxes(ticksuffix='%')
    return fig


def rating_to_stars(rating):
    if pd.isnull(rating):
        return ''
    full_stars = int(round(rating))
    return '⭐' * full_stars


if __name__ == '__main__':
    app.run_server(debug=True)
