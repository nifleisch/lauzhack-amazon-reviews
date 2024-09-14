import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd


extract_issues_system_prompt = '''
Your goal is to extract product issues from customer reviews of unlocked mobile phones.
You will be provided with a customer review, and you will output a json object containing the following information:

{
    issues: string[] // Array of issues extracted.
}

Each issue should be described in a simple and concise sentence. For example, "The battery life is too short." or "The camera quality is poor."
Only extract issues that are clearly mentioned in the review. If the review does not mention any issues, output an empty array.
'''


def extract_issues(review: str, client: OpenAI):
    """Extract issues from a review using the OpenAI API."""
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.1,
    response_format={
        "type": "json_object"
    },
    messages=[
        {
            "role": "system",
            "content": extract_issues_system_prompt
        },
        {
            "role": "user",
            "content": review
        }
    ],
    )
    return response.choices[0].message.content


def prepare_batch_tasks(reviews: list) -> list:
    tasks = []
    for (review_id, review) in reviews:
        task = {
            "custom_id": review_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "response_format": {
                    "type": "json_object"
                },
                "messages": [
                    {
                        "role": "system",
                        "content": extract_issues_system_prompt
                    },
                    {
                        "role": "user",
                        "content": review
                    }
                ],
            }
        }
        tasks.append(task)
    return tasks


if __name__ == "__main__":
    data_folder = Path('data')
    file_path = data_folder / 'subset.csv'
    df = pd.read_csv(file_path)
    reviews = list(zip(df.review_id, df.review))

    load_dotenv()
    open_ai_api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=open_ai_api_key)

    review_id, review = reviews[0]
    extracted_issues = extract_issues(review, client)

    batch_tasks = prepare_batch_tasks(reviews)
    file_name = "batch_tasks.jsonl"

    with open(data_folder / file_name, 'w') as f:
        for obj in batch_tasks:
            f.write(json.dumps(obj) + '\n')

    batch_file = client.files.create(
        file=open(data_folder / file_name, "rb"),
        purpose="batch"
        )

    batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
                )
