from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def regression_analysis(issue_filepath: Path, review_filepath: Path):
    issue_df = pd.read_csv(issue_filepath)
    cluster_names = issue_df.cluster_name.unique().tolist()
    issue_df = (issue_df
                .drop(columns=['issue_id', 'issue', 'emb_1', 'emb_2', 'cluster'])
                .pipe(pd.get_dummies, columns=['cluster_name'], dtype=float)
                .groupby('task_id')
                .max()
                .reset_index()
                )
    review_df = (pd.read_csv(review_filepath)
                 .drop(columns=['brand_name', 'review', 'n_review_votes'])
                 .assign(
                     product_name = lambda x: x.product_name.str.replace(r'[^a-zA-Z0-9_ ]', '', regex=True)
                 )
                 )

    product_names = review_df.product_name.unique().tolist()
    df = (review_df
          .merge(issue_df,
                 left_on='review_id',
                 right_on='task_id',
                 how='left')
          .drop(columns=['task_id', 'review_id', 'price'])
          .fillna(0.0)
          .pipe(pd.get_dummies, columns=['product_name'], dtype=float)
        )
    y = df.pop('rating')
    X = df

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Predict and evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f'Mean squared error: {mse}')
    print(f'R2 score: {r2}')
    make_coeff_plot(model.coef_, X.columns)

    results = []
    for product in product_names:
        product_col = f'product_name_{product}'
        product_df = df.query(f'`{product_col}` == 1').copy(deep=True)
        product_mean_pred = model.predict(product_df).mean()
        for issue in cluster_names:
            issue_col = f'cluster_name_{issue}'
            issue_col_vals = product_df[issue_col]
            # set issue col to 0
            product_df[issue_col] = 0
            issue_pred = model.predict(product_df).mean()
            results.append({
                'product_name': product,
                'issue': issue,
                'impact': product_mean_pred - issue_pred
            })
            # reset issue col
            product_df[issue_col] = issue_col_vals
    results_df = pd.DataFrame(results)
    return results_df


def issue_frequency_analysis(issue_filepath: Path, review_filepath: Path):
    issue_df = pd.read_csv(issue_filepath)
    cluster_names = issue_df.cluster_name.unique().tolist()
    issue_df = (issue_df
                .drop(columns=['issue_id', 'issue', 'emb_1', 'emb_2', 'cluster'])
                .pipe(pd.get_dummies, columns=['cluster_name'], dtype=float)
                .groupby('task_id')
                .max()
                .reset_index()
                )
    review_df = (pd.read_csv(review_filepath)
                 .drop(columns=['brand_name', 'review', 'n_review_votes'])
                 .assign(
                     product_name = lambda x: x.product_name.str.replace(r'[^a-zA-Z0-9_ ]', '', regex=True)
                 )
                 )

    product_names = review_df.product_name.unique().tolist()
    df = (review_df
          .merge(issue_df,
                 left_on='review_id',
                 right_on='task_id',
                 how='left')
          .drop(columns=['task_id', 'review_id', 'price'])
          .fillna(0.0)
          .pipe(pd.get_dummies, columns=['product_name'], dtype=float)
        )

    results = []

    for issue in cluster_names:
        issue_col = f'cluster_name_{issue}'
        issue_frequency = df[issue_col].mean()
        for product in product_names:
            product_col = f'product_name_{product}'
            product_df = df.query(f'`{product_col}` == 1').copy(deep=True)
            product_issue_frequency = product_df[issue_col].mean()
            results.append({
                'product_name': product,
                'issue': issue,
                'issue_frequency': issue_frequency,
                'product_issue_frequency': product_issue_frequency,
                'frequency_difference': product_issue_frequency - issue_frequency
            })
    results_df = pd.DataFrame(results)
    return results_df


def combine_results(issue_filepath, review_filepath):
    issue_df = pd.read_csv(issue_filepath)
    review_df = pd.read_csv(review_filepath)
    combined_df = (review_df
                   .merge(issue_df,
                          left_on='review_id',
                          right_on='task_id',
                          how='left')
                   .assign(
                       product_name = lambda x: x.product_name.str.replace(r'[^a-zA-Z0-9_ ]', '', regex=True)
                    )
                     .drop(columns=['brand_name', 'price', 'cluster'])
                     .query('~cluster_name.isna()')
                   )
    return combined_df


def make_coeff_plot(coef, columns, n=20):
    columns = [col.split('_')[-1] for col in columns]
    coefficients = pd.Series(coef, index=columns)
    most_negative_coeffs = coefficients.sort_values().head(n)
    most_negative_coeffs = most_negative_coeffs * -1
    plt.figure(figsize=(10, 6))
    sns.barplot(x=most_negative_coeffs.values, y=most_negative_coeffs.index)
    plt.xlabel('Drop in Rating')
    plt.ylabel('')
    plot_folder = Path('assets')
    plot_folder.mkdir(exist_ok=True)
    plt.savefig(plot_folder / 'coeffs.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    data_folder = Path('data')
    issue_filepath = data_folder / 'issue.csv'
    review_filepath = data_folder / 'subset.csv'
    regression_df = regression_analysis(issue_filepath, review_filepath)
    regression_df.to_csv(data_folder / 'regression_results.csv', index=False)
    issue_fequency_df = issue_frequency_analysis(issue_filepath, review_filepath)
    issue_fequency_df.to_csv(data_folder / 'issue_frequency_results.csv', index=False)
    combined_df = combine_results(issue_filepath, review_filepath)
    combined_df.to_csv(data_folder / 'combined_results.csv', index=False)
