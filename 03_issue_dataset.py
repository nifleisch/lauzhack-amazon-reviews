from pathlib import Path
import uuid
import json
from typing import List, Dict

import pandas as pd


def load_results(filepath: Path) -> pd.DataFrame:
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results


def make_issue_dataset(results: List[Dict]) -> pd.DataFrame:
    issue_list = []
    for result in results:
        task_id = result['custom_id']
        result = json.loads(result['response']['body']['choices'][0]['message']['content'])
        issues = result['issues']
        for issue in issues:
            issue_list.append({
                'task_id': task_id,
                'issue_id': uuid.uuid4(),
                'issue': issue
            })
    return pd.DataFrame(issue_list)


if __name__ == "__main__":
    data_folder = Path('data')
    filepath = data_folder / 'batch_output.jsonl'
    results = load_results(filepath)
    df = make_issue_dataset(results)
    df.to_csv(data_folder / 'issue.csv', index=False)
