import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('archive/scraped.csv')
    df = df.drop(0)
    df.to_csv('ellie_scraped.csv')