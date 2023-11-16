import pandas as pd
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests as r
import time
from tqdm import tqdm


# CLIENT_ID = "662e93abaeff48e5bc0d755259cdb707"
# CLIENT_SECRET = "60c642c3f153486a84e0605b35bf50dd"


def get_midi_spotify_track_data(sids):
    midi_data = []
    cid ="662e93abaeff48e5bc0d755259cdb707"
    secret = "60c642c3f153486a84e0605b35bf50dd"
    client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    #spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

    #create file 
    temp = {'track_id': 0, 'track_name': '', 'track_artist' : '', 'lyrics' : '', 'track_popularity' : 0, 'track_album_id' : 0, 'track_album_name' : '', 'track_album_release_date' : 0, 'playlist_name' : None, 'playlist_id' : None, 'playlist_genre' : None, 'playlist_subgenre' : None, 'language' : '', 'danceability':0,
    'energy':0,'key':0,'loudness':0, 'mode':0,'speechiness':0, 'acousticness':0,'instrumentalness':0, 'liveness':0,'valence':0,'tempo':0, 'duration_ms':0}
    pd.DataFrame.from_dict([temp]).to_csv('scraped.csv', index=False)

    i = 0
    for track_id in tqdm(sids):
        
        # get meta_data
        meta_data = spotify.track(track_id)
        track_name = meta_data['name']
        track_artist = meta_data['artists'][0]['name']
        track_popularity = meta_data['popularity']
        track_album_id = meta_data['album']['id']
        track_album_name = meta_data['album']['name']
        track_album_release_date = meta_data['album']['release_date']

        # lyrics
        lyrics = None
        language = None
        link = f"https://spotify-lyric-api-984e7b4face0.herokuapp.com/?trackid={track_id}"
        results = r.get(link).json()
        if results['error'] == False:
            lyrics = ""
            for obj in results['lines']:
                lyrics += obj['words']
                lyrics += " "
            lyrics = lyrics[:-1]
            language = 'en'
            
        parameters = {'track_id': track_id, 'track_name': track_name, 'track_artist' : track_artist, 'lyrics' : lyrics, 'track_popularity' : track_popularity, 'track_album_id' : track_album_id, 'track_album_name' : track_album_name, 'track_album_release_date' : track_album_release_date, 'playlist_name' : None, 'playlist_id' : None, 'playlist_genre' : None, 'playlist_subgenre' : None, 'language' : language}
        audio_features = spotify.audio_features(track_id)
        parameters['danceability'] = audio_features[0]['danceability']
        parameters['energy'] = audio_features[0]['energy']
        parameters['key'] = audio_features[0]['key']
        parameters['loudness'] = audio_features[0]['loudness']
        parameters['mode'] = audio_features[0]['mode']
        parameters['speechiness'] = audio_features[0]['speechiness']
        parameters['acousticness'] = audio_features[0]['acousticness']
        parameters['instrumentalness'] = audio_features[0]['instrumentalness']
        parameters['liveness'] = audio_features[0]['liveness']
        parameters['valence'] = audio_features[0]['valence']
        parameters['tempo'] = audio_features[0]['tempo']
        parameters['duration_ms'] = audio_features[0]['duration_ms']
        # compile
        
        # parameters = {'track_id': track_id, 'track_name': track_name, 'track_artist' : track_artist, 'lyrics' : lyrics, 'track_popularity' : track_popularity, 'track_album_id' : track_album_id, 'track_album_name' : track_album_name, 'track_album_release_date' : track_album_release_date, 'playlist_name' : None, 'playlist_id' : None, 'playlist_genre' : None, 'playlist_subgenre' : None, 'danceability' : danceability, 'energy' : energy, 'key' : key, 'loudness' : loudness, 'mode' : mode, 'speechiness' : speechiness, 'acousticness' : acousticness, 'instrumentalness' : instrumentalness, 'liveness' : liveness, 'valence' : valence, 'tempo' : tempo, 'duration_ms' : duration_ms, 'language' : language}
        midi_data.append(parameters)
        #flush every 500 sids
        if i== 500:
            print(i)
            pd.DataFrame(midi_data).to_csv('scraped.csv', mode = 'a', header=None, index=False)
            i = 0
            midi_data = []
        i +=1
    return midi_data


def get_midi_spotify_audio_data(midi_data):
    try:
        spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

        for parameters in midi_data:
            track_id = parameters['track_id']
            print(track_id)
            audio_features = spotify.audio_features(track_id)
            parameters['danceability'] = audio_features[0]['danceability']
            parameters['energy'] = audio_features[0]['energy']
            parameters['key'] = audio_features[0]['key']
            parameters['loudness'] = audio_features[0]['loudness']
            parameters['mode'] = audio_features[0]['mode']
            parameters['speechiness'] = audio_features[0]['speechiness']
            parameters['acousticness'] = audio_features[0]['acousticness']
            parameters['instrumentalness'] = audio_features[0]['instrumentalness']
            parameters['liveness'] = audio_features[0]['liveness']
            parameters['valence'] = audio_features[0]['valence']
            parameters['tempo'] = audio_features[0]['tempo']
            parameters['duration_ms'] = audio_features[0]['duration_ms']
    except Exception as e:
        print("Gathering all data failed due to:" + str(e))
    df = pd.DataFrame(midi_data)
    return df


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
    # filtered_df.to_csv('dat/filtered_data.csv')
    return filtered_df

def add_midi_association(midi_df, filtered_df):
    """
    using midi_df and filtered_df, find the midis associated with sid
    """
    print(filtered_df)
    merged_df = filtered_df.merge(midi_df, left_on='track_id', right_on='sid', how='left')
    merged_df = merged_df.drop('sid', axis=1)
    # merged_df.to_csv('dat/filtered_data.csv')
    return merged_df


if __name__ == "__main__":
    midi_tsv = 'dat/midi/MMD_audio_text_matches.tsv'
    spotify_metadata = 'dat/lyrics/spotify_songs.csv'

    midi_df = pd.read_csv(midi_tsv, sep='\t')
    spotify_df = pd.read_csv(spotify_metadata)

    sids = find_overlap(midi_df, spotify_df)
    spotify_metadata_df = get_spotify_metadata_csv(spotify_df, sids)
    # print(midi_df)
    filtered_df = add_midi_association(midi_df, spotify_metadata_df)

    # midi_sids = find_disjoint(midi_df, spotify_df)

    # midi_track_data = get_midi_spotify_track_data(midi_sids)
    # midi_full_df = get_midi_spotify_audio_data(midi_track_data)
    #
    # cumul_df = pd.concat([spotify_metadata_df, midi_full_df])
    # #
    # cumul_df.to_csv('dat/TEMP_all_midi_metadata.csv')

