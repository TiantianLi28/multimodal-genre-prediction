import pandas as pd
import numpy as np
import time
import json
import base64
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests as r
import time
from tqdm import tqdm
from requests import post,get

cid = "0b0e99672e2a4a06a469fc05ddf53906"
secret = "44a9ae8f86c643a3abeb22250af40fb4"

def get_token():
    auth_string = cid + ":" + secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes),"utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization":"Basic " + auth_base64,
        "Content-Type":"application/x-www-form-urlencoded"
    }
    data = {"grant_type":'client_credentials'}
    result = post(url,headers=headers,data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization":"Bearer " + token}

def get_midi_spotify_audio_features_wspotify(token,marker,midis,sids):
    midi_data = []
    query = "?ids="
    count = 0
    for sid in sids:
        if(count < (len(sids) -1) ):
            query = query + f"{sid},"
        else:
            query = query + f"{sid}"
        count += 1

    url1 = "https://api.spotify.com/v1/audio-features/"
    query_url1 = url1 + query
    headers = get_auth_header(token)
    # print(query_url1)
    json_audio = get(query_url1,headers=headers)
    # print(json_audio)
    audio_features = json.loads(json_audio.content)
    audio_features_data = audio_features["audio_features"]
    # print(audio_features_data)
    # print(len(meta_data["tracks"]))

    for track in audio_features_data:
        track_id = track['id']
        track_danceability = track['danceability']
        track_energy = track['energy']
        track_key = track['key']
        track_loudness = track['loudness']
        track_mode = track['mode']
        track_speechiness = track['speechiness']
        track_acousticness = track['acousticness']
        track_instrumentalness = track['instrumentalness']
        track_liveness = track['liveness']
        track_valence = track['valence']
        track_tempo = track['tempo']
        track_duration_ms = track['duration_ms']
        # parameters['md5'] = midis[i]
        # check out what md5 is 
      

       

        parameters = {'track_id': track_id,'danceability':track_danceability,'energy':track_energy,'key':track_key,'loudness':track_loudness,'mode':track_mode,'speechiness':track_speechiness,'acousticness':track_acousticness,'instrumentalness':track_instrumentalness,'liveness':track_liveness,'valence':track_valence,'tempo':track_tempo,'duration_ms':track_duration_ms}
        midi_data = [parameters]
        pd.DataFrame(midi_data).to_csv('scraped_features.csv', mode='a', header=None, index=False)
    
    return midi_data

def get_midi_spotify_track_data_wspotify(token,marker, midis, sids):
    midi_data = []


    query = "?ids="
    count = 0
    for sid in sids:
        if(count < (len(sids) -1) ):
            query = query + f"{sid},"
        else:
            query = query + f"{sid}"
        count += 1

    url1 = "https://api.spotify.com/v1/tracks/"
    query_url1 = url1 + query
    headers = get_auth_header(token)
    # print(query_url1)
    json_meta = get(query_url1,headers=headers)
    # print(json_meta)
    meta_data = json.loads(json_meta.content)
    track_meta_data = meta_data["tracks"]
    # print(len(meta_data["tracks"]))

    for track in track_meta_data:
        track_id = track['id']
        track_name = track['name']
        track_artist = track['artists'][0]['name']
        track_popularity = track['popularity']
        track_album_id = track['album']['id']
        track_album_name = track['album']['name']
        track_album_release_date = track['album']['release_date']

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
        midi_data = [parameters]
        pd.DataFrame(midi_data).to_csv('scraped_tracks.csv', mode='a', header=None, index=False)
        
    return midi_data


def join_track_audio_csv():
    track_columns=['track_id','track_name','track_artist','lyrics','track_popularity','track_album_id','track_album_name','track_album_release_date','playlist_name','playlist_id','playlist_genre','playlist_subgenre','language']
    # df_track = pd.DataFrame(columns=['track_id','track_name','track_artist','lyrics','track_popularity','track_album_id','track_album_name','track_album_release_date','playlist_name','playlist_id','playlist_genre','playlist_subgenre','language']) 
    # import csv with data 

    track_csv = 'scraped_tracks.csv'
    track_data_append = pd.read_csv(track_csv,header=None,names=track_columns)

    # print(track_data_append.head)

    # track_results_df = pd.concat([df_track,track_data_append],axis=1,ignore_index=True)

    # track_results_df.head

    # create dataframe with appropriate headers for audio features 
    audio_columns=['track_id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']
    # df_audio = pd.DataFrame(columns=['track_id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms']) 
    # import csv with audio data

    audio_csv = 'scraped_features.csv'
    audio_data_append = pd.read_csv(audio_csv,header=None,names=audio_columns)

    # print(audio_data_append.head)
    # audio_results_df = pd.concat([df_audio,audio_data_append],axis=1)

    new_column_names = {'md5': 'md5', 'score': 'score','sid':"track_id"}
    midi_df_csv = 'midi_df_israel.csv'
    midi_df_data = pd.read_csv(midi_df_csv)
    midi_df_data.rename(columns=new_column_names, inplace=True)
    midi_df_data.drop('score', axis=1, inplace=True)


    #concatenate on the right column 
    merged_track_audio = track_data_append.merge(audio_data_append,how='inner',on='track_id')

    merged_track_audio = merged_track_audio.merge(midi_df_data,how='inner',on='track_id')

    merged_track_audio.drop_duplicates(subset="track_id", keep='first', inplace=True)

    # merged_track_audio.head

    #reorder to match already existing format 

    merged_track_audio.to_csv('scrapes_concat.csv')