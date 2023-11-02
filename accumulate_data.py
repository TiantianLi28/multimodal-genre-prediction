import pandas as pd
import numpy as np


def find_overlap(midi_df, spotify_df):
    """
    for extracting overlapping Spotify IDs from two datasets
    """
    midi_sids = set(midi_df['sid'])
    spotify_sids = set(spotify_df['track_id'])

    cumul_sids = list(set(midi_sids) & set(spotify_sids))

    return cumul_sids

def get_spotify_metadata_csv(spotify_df, cumul_sids):
    """
    using the sids extracted, separate metadata from cumulative spotify data
    """
    filtered_df = spotify_df[spotify_df['track_id'].isin(cumul_sids)]
    filtered_df.to_csv('dat/filtered_data.csv')


if __name__ == "__main__":
    midi_tsv = 'dat/midi/MMD_audio_text_matches.tsv'
    spotify_metadata = 'dat/lyrics/spotify_songs.csv'

    midi_df = pd.read_csv(midi_tsv, sep='\t')
    spotify_df = pd.read_csv(spotify_metadata)

    sids = find_overlap(midi_df, spotify_df)
    get_spotify_metadata_csv(spotify_df, sids)
