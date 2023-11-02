import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# CLIENT_ID = "662e93abaeff48e5bc0d755259cdb707"
# CLIENT_SECRET = "60c642c3f153486a84e0605b35bf50dd"


def get_midi_spotify_data(sids):
    # parameters = {'track_id' : None ,'track_name','track_artist','lyrics','track_popularity','track_album_id','track_album_name','track_album_release_date','playlist_name','playlist_id','playlist_genre','playlist_subgenre','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','language'}
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
    # for id in sids:


    meta_data = spotify.track(sids[0])
    audio_features = spotify.audio_analysis(sids[0])
    return meta_data, audio_features

def find_overlap(midi_df, spotify_df):
    """
    for extracting overlapping Spotify IDs from two datasets
    """
    midi_sids = set(midi_df['sid'])
    spotify_sids = set(spotify_df['track_id'])

    cumul_sids = list(set(midi_sids) & set(spotify_sids))

    return cumul_sids

def find_disjoint(midi_df, cumul_sids):
    midi_sids = set(midi_df['sid'])

    midi_only = [id for id in midi_sids if id not in cumul_sids]

    return midi_only

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
    # get_spotify_metadata_csv(spotify_df, sids)

    midi_sids = find_disjoint(midi_df, spotify_df)

    print(get_midi_spotify_data(midi_sids))