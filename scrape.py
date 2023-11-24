import pandas as pd
import numpy as np
import os
from accumulate_data2 import get_midi_spotify_audio_features_wspotify,get_midi_spotify_track_data_wspotify,get_token,join_track_audio_csv

# create file in the beginning
# temp = {'track_id': 0, 'track_name': '', 'track_artist' : '', 'lyrics' : '', 'track_popularity' : 0, 'track_album_id' : 0, 'track_album_name' : '', 'track_album_release_date' : 0, 'playlist_name' : None, 'playlist_id' : None, 'playlist_genre' : None, 'playlist_subgenre' : None, 'language' : '', 'danceability':0,
# 'energy':0,'key':0,'loudness':0, 'mode':0,'speechiness':0, 'acousticness':0,'instrumentalness':0, 'liveness':0,'valence':0,'tempo':0, 'duration_ms':0, 'md5' : ''}
# pd.DataFrame.from_dict([temp]).to_csv('scraped.csv', index=False)

# path = 'midi_df_tiantian.csv'
# path = 'midi_df_ellie.csv'
path = 'midi_df_israel.csv'
midi_df = pd.read_csv(path)
midis = list(midi_df['md5'])
sids = list(midi_df['sid'])



#this is for scraping track data
# sids_len = len(sids)
# j = 48033 #this is used to handle where it stops 
# no_of_sids = j + 40
# tsids = sids[j:no_of_sids]
# token=get_token()
# while(j<sids_len):
#     tsids = sids[j:no_of_sids]
#     get_midi_spotify_track_data_wspotify(token,j,midi_df,tsids)
#     j += 40
#     no_of_sids += 40

#this is for scraping audio features 
# sids_len = len(sids)
# j = 80001 #this is used to handle where it stops 
# no_of_sids = j + 40
# tsids = sids[j:no_of_sids]
# token=get_token()
# while(j<sids_len):
#     tsids = sids[j:no_of_sids]
#     get_midi_spotify_audio_features_wspotify(token,j,midi_df,tsids)
#     j += 40
#     no_of_sids += 40


#this joins the two csv files on track_id after scraping
join_track_audio_csv()