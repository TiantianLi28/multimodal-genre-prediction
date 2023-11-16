import pandas as pd
import numpy as np
import os
from accumulate_data import get_midi_spotify_track_data

path = 'midi_df_tiantian.csv'
# path = 'midi_df_ellie.csv'
# path = 'midi_df_israel.csv'
midi_df = pd.read_csv(path)
get_midi_spotify_track_data(midi_df['sid'])