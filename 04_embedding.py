import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import openTSNE
import pandas as pd


def get_embeddings(
    list_of_text: List[str],
    client: OpenAI,
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
) -> List[List[float]]:
    data = client.embeddings.create(input=list_of_text,
                                    model=model,
                                    dimensions=dimensions).data
    return [d.embedding for d in data]


def embed_issues(issue_list: List[str], client: OpenAI, chunk_size: int = 2000):
    chunk_size = 2000
    embeddings = []
    chunks = [issue_list[i:i + chunk_size] for i in range(0, len(issue_list), chunk_size)]
    for chunk in chunks:
        embeddings += get_embeddings(chunk, client)
    return embeddings


def t_sne(embeddings: List[List[float]]):
    x = np.array(embeddings)
    affinities_multiscale_mixture = openTSNE.affinity.Multiscale(
                                        x,
                                        perplexities=[50, 500],
                                        metric="cosine",
                                        n_jobs=8,
                                        random_state=3,
                                    )
    init = openTSNE.initialization.pca(x, random_state=42)
    embedding_multiscale = openTSNE.TSNE(n_jobs=8).fit(
                                        affinities=affinities_multiscale_mixture,
                                        initialization=init,
                                    )
    return embedding_multiscale


if __name__ == "__main__":
    data_folder = Path('data')
    filepath = data_folder / 'issue.csv'
    df = pd.read_csv(filepath)

    load_dotenv()
    open_ai_api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=open_ai_api_key)

    issue_list = df.issue.tolist()
    embeddings = embed_issues(issue_list, client)

    embedding = t_sne(embeddings)
    df['emb_1'] = embedding[:, 0]
    df['emb_2'] = embedding[:, 1]
    df.to_csv(data_folder / 'issue.csv', index=False)
