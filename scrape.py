import pandas as pd
from dat.accumulate_data import get_midi_spotify_track_data

# create file in the beginning
# temp = {'track_id': 0, 'track_name': '', 'track_artist' : '', 'lyrics' : '', 'track_popularity' : 0, 'track_album_id' : 0, 'track_album_name' : '', 'track_album_release_date' : 0, 'playlist_name' : None, 'playlist_id' : None, 'playlist_genre' : None, 'playlist_subgenre' : None, 'language' : '', 'danceability':0,
# 'energy':0,'key':0,'loudness':0, 'mode':0,'speechiness':0, 'acousticness':0,'instrumentalness':0, 'liveness':0,'valence':0,'tempo':0, 'duration_ms':0, 'md5' : ''}
# pd.DataFrame.from_dict([temp]).to_csv('scraped.csv', index=False)

# path = 'midi_df_tiantian.csv'
path = 'dat/midi_to_scrape/midi_df_ellie.csv'
# path = 'midi_df_israel.csv'
midi_df = pd.read_csv(path)
midis = list(midi_df['md5'])
sids = list(midi_df['sid'])

i = 16073
sids = sids[i:]
# print(sids)
get_midi_spotify_track_data(i, midis, sids)