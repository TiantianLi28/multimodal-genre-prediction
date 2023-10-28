import pandas as pd
import numpy as np

"""
for extracting overlapping Spotify IDs from two datasets
"""

if __name__ == "__main__":
    midi_tsv = 'dat/midi/MMD_audio_text_matches.tsv'
    spotify_metadata = 'dat/lyrics/spotify_songs.csv'

    midi_df = pd.read_csv(midi_tsv, sep='\t')
    spotify_df = pd.read_csv(spotify_metadata)

    midi_sids = set(midi_df['sid'])
    spotify_sids = set(spotify_df['track_id'])

    cumul_sids = list(set(midi_sids) & set(spotify_sids))

    print(cumul_sids)
    print(len(cumul_sids))
