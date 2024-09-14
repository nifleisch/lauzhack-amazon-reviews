from pathlib import Path
import uuid

import pandas as pd


def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.rename(columns={
        'Product Name': 'product_name',
        'Brand Name': 'brand_name',
        'Price': 'price',
        'Rating': 'rating',
        'Reviews': 'review',
        'Review Votes': 'n_review_votes'
    }, inplace=True)
    df['review_id'] = df.apply(lambda x: uuid.uuid4(), axis=1)
    return df


def make_subset_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Make a subset of the data to reduce LLM cost. Subset the following way:
    1. Only consider the 100 most reviewed products.
    2. Eliminate duplicate reviews for each product.
    3. Only consider reviews with more than 40 characters (~ 1 sentence).
    """
    most_reviewed_products = df.product_name.value_counts().head(100).index
    df_list = []
    for product in most_reviewed_products:
        product_df = (df
                      .query('product_name == @product')
                      .query('review.str.len() >= 40')
                      .drop_duplicates(subset='review')
                     )
        df_list.append(product_df)
    return pd.concat(df_list).reset_index(drop=True)


if __name__ == "__main__":
    data_folder = Path('data')
    filepath = data_folder / 'Amazon_Unlocked_Mobile.csv'
    df = load_data(filepath)
    subset_df = make_subset_df(df)
    subset_df.to_csv(data_folder / 'subset.csv', index=False)
