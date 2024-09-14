import json
import os
from pathlib import Path

from dotenv import load_dotenv
import plotly.express as px
import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.cluster import Birch


name_cluster_system_prompt = '''
Your goal is to assign a concise name to a list of sentences that all describe the same issue with a mobile phone.
You will be provided with a list of sentences, and you will output a json object containing the following information:

{
    issue: string // String that describes the issue with at most 3 words.
}

For example, the sentences "The battery life is too short." and "The battery only lasts 2 hours." could be described as "Short battery life".
'''


def db_scan(df: pd.DataFrame):
    x_1, x_2 = df.emb_1, df.emb_2
    X = np.array(list(zip(x_1, x_2)))
    dbscan = Birch(n_clusters=50)
    return dbscan.fit_predict(X)


def name_cluster(examples: str, client: OpenAI):
    response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.1,
    response_format={
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": name_cluster_system_prompt
        },
        {
            "role": "user",
            "content": examples
        }
    ],
    )
    result = json.loads(response.choices[0].message.content)
    return result['issue']


def name_clusters(df: pd.DataFrame, client: OpenAI):
    cluster_names = {-1: 'Other'}
    for cluster in range(df.cluster.max() + 1):
        examples = df.query(f'cluster == {cluster}').issue.head(10).tolist()
        examples = '\n'.join(examples)
        cluster_names[cluster] = name_cluster(examples, client)
    df['cluster_name'] = df.cluster.map(cluster_names)
    return df


if __name__ == "__main__":
    data_folder = Path('data')
    filepath = data_folder / 'issue.csv'
    df = pd.read_csv(filepath)
    df['cluster'] = db_scan(df)

    # plot clusters to tune parameters
    #fig = px.scatter(df, x="emb_1", y="emb_2", color="cluster",
    #             hover_data=['issue'])
    #fig.show()

    load_dotenv()
    open_ai_api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=open_ai_api_key)

    df = name_clusters(df, client)
    df.to_csv(data_folder / 'issue.csv', index=False)
