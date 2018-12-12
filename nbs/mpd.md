

```python
import numpy as np
import pandas as pd
import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import os
import pickle
import json
```


```python
num_playlists = 0
n = 2000
#wanted_album = "spotify:album:6vV5UrXcfyQD1wu4Qo2I9K"
my_playlists = []
for filename in os.listdir('data'):
    a = open('data/' + filename).read()
    temp_data = json.loads(a)
    for playlist in temp_data["playlists"]:
        if num_playlists >=n:
            break
        if playlist["num_followers"] > 10000:
            num_playlists += 1
            my_playlists.append(playlist)
        #for track in playlist["tracks"]:
            #if track["album_uri"] == wanted_album:
                #num_playlists += 1
                #my_playlists.append(playlist)
                #continue
```


```python
#my_playlists[1999]
```


```python
def refresh():
    os.environ["SPOTIPY_CLIENT_ID"] = '4a7c50d7174a4a66a4cd613dee01fcd9'
    os.environ["SPOTIPY_CLIENT_SECRET"] = 'fcc804b6e96e454f9579666137df3585'
    os.environ["SPOTIPY_REDIRECT_URI"] = 'http://localhost:8888/callback'

    scope = 'user-library-read'

    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        #print("Usage: %s username" % (sys.argv[0],))
        sys.exit()

    token = util.prompt_for_user_token(username, scope)

    # Print some tracks
    if token:
        sp = spotipy.Spotify(auth=token)
        results = sp.current_user_saved_tracks()
        for item in results['items']:
            track = item['track']
            #print(track['name'] + ' - ' + track['artists'][0]['name'])
    else:
        print("Can't get token for", username)
    return sp

sp = refresh()
```


```python
def get_song_level_data(spotify_ids):
    '''
    Given a list of spotify IDs, get the popularity and market info for the songs
    '''
    chunk_size= 42
    tmp = {}
    for i in range(0, len(spotify_ids), chunk_size):
        chunk = spotify_ids[i:i+chunk_size]
        features = sp.tracks(chunk)['tracks']
        features  = pd.DataFrame([x for x in features if isinstance(x, dict)])
        tmp_df = pd.DataFrame(features)
        tmp_df.index = tmp_df['id']
        #tmp_df = tmp_df[['id', 'popularity']]
        tmp.update(tmp_df.T.to_dict())
    df = pd.DataFrame(tmp).T
    return df
```


```python
playlist_data = {}
tracks_in_playlists = {}
for playlist in my_playlists:
    i = playlist["pid"]
    list_of_ids = []
    for track in playlist["tracks"]:
        list_of_ids.append(track["track_uri"])
    tracks_in_playlists[i] = list_of_ids
    popularities = get_song_level_data(list_of_ids)
    mean_popularity = np.mean(popularities["popularity"])
    #print(popularities)
    playlist_data[i] = {}
    playlist_data[i]["popularity"] = mean_popularity
    playlist_data[i]["num_followers"] = playlist["num_followers"]
```

    C:\Users\Elbert\Anaconda3\lib\site-packages\ipykernel_launcher.py:14: UserWarning: DataFrame columns are not unique, some columns will be omitted.
      
    


```python
len(tracks_in_playlists)
```




    17




```python
with open('tracks_in_playlists2.p', 'wb') as fp:
    pickle.dump(tracks_in_playlists, fp, protocol=pickle.HIGHEST_PROTOCOL)
```


```python
with open('playlist_data2.p', 'wb') as fp:
    pickle.dump(playlist_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
```


```python
my_playlists[0]
```




    {'collaborative': 'false',
     'duration_ms': 5900266,
     'modified_at': 1456531200,
     'name': 'Workout Playlist ',
     'num_albums': 23,
     'num_artists': 19,
     'num_edits': 8,
     'num_followers': 11745,
     'num_tracks': 26,
     'pid': 101121,
     'tracks': [{'album_name': 'Break a Sweat',
       'album_uri': 'spotify:album:2nWxnJlsL1bzC2wHtouEvZ',
       'artist_name': 'Becky G',
       'artist_uri': 'spotify:artist:4obzFoKoKRHIphyHzJ35G3',
       'duration_ms': 208240,
       'pos': 0,
       'track_name': 'Break a Sweat',
       'track_uri': 'spotify:track:28r9sD2b6FluK7YHjTJ0fl'},
      {'album_name': 'Purpose',
       'album_uri': 'spotify:album:7fZH0aUAjY3ay25obOUf2a',
       'artist_name': 'Justin Bieber',
       'artist_uri': 'spotify:artist:1uNFoZAHBGtllmzznpCI3s',
       'duration_ms': 200786,
       'pos': 1,
       'track_name': 'Sorry',
       'track_uri': 'spotify:track:69bp2EbF7Q2rqc5N3ylezZ'},
      {'album_name': '4',
       'album_uri': 'spotify:album:1gIC63gC3B7o7FfpPACZQJ',
       'artist_name': 'Beyoncé',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'duration_ms': 236093,
       'pos': 2,
       'track_name': 'Run the World (Girls)',
       'track_uri': 'spotify:track:1uXbwHHfgsXcUKfSZw5ZJ0'},
      {'album_name': 'Right Hand',
       'album_uri': 'spotify:album:6qIBaopvx9STCAa7ud8PRZ',
       'artist_name': 'Drake',
       'artist_uri': 'spotify:artist:3TVXtAsR1Inumwj472S9r4',
       'duration_ms': 190595,
       'pos': 3,
       'track_name': 'Right Hand',
       'track_uri': 'spotify:track:3lSR267IJfT54p0Gfuw7mi'},
      {'album_name': 'Peace Is The Mission',
       'album_uri': 'spotify:album:2XBnxKeRZi76u2iyGcMych',
       'artist_name': 'Major Lazer',
       'artist_uri': 'spotify:artist:738wLrAtLtCtFOLvQBXOXp',
       'duration_ms': 176561,
       'pos': 4,
       'track_name': 'Lean On (feat. MØ & DJ Snake)',
       'track_uri': 'spotify:track:4WjH9Bzt3kx7z8kl0awxh4'},
      {'album_name': 'Good For You',
       'album_uri': 'spotify:album:5TXJJMTib9nAPLKPM1PrMd',
       'artist_name': 'Selena Gomez',
       'artist_uri': 'spotify:artist:0C8ZW7ezQVs4URX5aX7Kqx',
       'duration_ms': 247133,
       'pos': 5,
       'track_name': 'Good For You - Phantoms Remix',
       'track_uri': 'spotify:track:2yGkxiZAitlsjNhqUmEm8U'},
      {'album_name': 'Hey Mama (feat. Nicki Minaj, Bebe Rexha & Afrojack)',
       'album_uri': 'spotify:album:3lLAW5J5IKH4SbENkAgRJT',
       'artist_name': 'David Guetta',
       'artist_uri': 'spotify:artist:1Cs0zKBU1kc0i8ypK3B9ai',
       'duration_ms': 171280,
       'pos': 6,
       'track_name': 'Hey Mama (feat. Nicki Minaj, Bebe Rexha & Afrojack) - Boaz van de Beatz remix',
       'track_uri': 'spotify:track:5GbQRYPXgTxONbGM92fxYs'},
      {'album_name': 'I Love Dance, Vol. 28',
       'album_uri': 'spotify:album:58s2dXgF6cUAdfxKV1kaq6',
       'artist_name': 'Java',
       'artist_uri': 'spotify:artist:5P9gekewyiWxrNSpGDao43',
       'duration_ms': 279173,
       'pos': 7,
       'track_name': 'Booty - R.P. Mix',
       'track_uri': 'spotify:track:5wpqE6kYHEvNxeyKvYwkKQ'},
      {'album_name': 'BEYONCÉ [Platinum Edition]',
       'album_uri': 'spotify:album:2UJwKSBUz6rtW4QLK74kQu',
       'artist_name': 'Beyoncé',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'duration_ms': 234413,
       'pos': 8,
       'track_name': 'Flawless Remix',
       'track_uri': 'spotify:track:0zVMzJ37VQNFUNvdxxat2E'},
      {'album_name': 'Britney Jean (Deluxe Version)',
       'album_uri': 'spotify:album:5rlB2HPoNHg2m1wmmh0TRv',
       'artist_name': 'Britney Spears',
       'artist_uri': 'spotify:artist:26dSoYclwsYLMAKD3tpOr4',
       'duration_ms': 247960,
       'pos': 9,
       'track_name': 'Work B**ch',
       'track_uri': 'spotify:track:3KliPMvk1EvFZu9cvkj8p1'},
      {'album_name': 'What Now',
       'album_uri': 'spotify:album:3jZh9gUtowy5cZlx5I1bfU',
       'artist_name': 'Rihanna',
       'artist_uri': 'spotify:artist:5pKCCKE2ajJHZ9KAiaK11H',
       'duration_ms': 291506,
       'pos': 10,
       'track_name': 'What Now - R3hab Remix',
       'track_uri': 'spotify:track:5VPW3rqpdGejKr79LHgjbg'},
      {'album_name': 'Summertime Sadness [Lana Del Rey vs. Cedric Gervais]',
       'album_uri': 'spotify:album:3GljjAP9QIKRQXeKoXHRnH',
       'artist_name': 'Lana Del Rey',
       'artist_uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa',
       'duration_ms': 214912,
       'pos': 11,
       'track_name': 'Summertime Sadness [Lana Del Rey vs. Cedric Gervais] - Cedric Gervais Remix',
       'track_uri': 'spotify:track:6D5pfooPP6hi99RaXjkDsP'},
      {'album_name': 'Habits (Stay High)',
       'album_uri': 'spotify:album:0JJgffTDTaNEYNKUhckZOc',
       'artist_name': 'Tove Lo',
       'artist_uri': 'spotify:artist:4NHQUGzhtTLFvgF5SZesLK',
       'duration_ms': 294253,
       'pos': 12,
       'track_name': 'Habits (Stay High) - The Chainsmokers Extended Mix',
       'track_uri': 'spotify:track:0n9iqBDje39F71sJxCzKLl'},
      {'album_name': 'BEYONCÉ [Platinum Edition]',
       'album_uri': 'spotify:album:2UJwKSBUz6rtW4QLK74kQu',
       'artist_name': 'Beyoncé',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'duration_ms': 213506,
       'pos': 13,
       'track_name': '7/11',
       'track_uri': 'spotify:track:02M6vucOvmRfMxTXDUwRXu'},
      {'album_name': 'BEYONCÉ [Platinum Edition]',
       'album_uri': 'spotify:album:2UJwKSBUz6rtW4QLK74kQu',
       'artist_name': 'Beyoncé',
       'artist_uri': 'spotify:artist:6vWDO969PvNqNYHIOW5v0m',
       'duration_ms': 323480,
       'pos': 14,
       'track_name': 'Drunk in Love',
       'track_uri': 'spotify:track:6jG2YzhxptolDzLHTGLt7S'},
      {'album_name': 'DNCE',
       'album_uri': 'spotify:album:7K89F9bgY1jks0uIlMerm3',
       'artist_name': 'DNCE',
       'artist_uri': 'spotify:artist:6T5tfhQCknKG4UnH90qGnz',
       'duration_ms': 219146,
       'pos': 15,
       'track_name': 'Cake By The Ocean',
       'track_uri': 'spotify:track:2aFiaMXmWsM3Vj72F9ksBl'},
      {'album_name': 'Blurryface',
       'album_uri': 'spotify:album:3cQO7jp5S9qLBoIVtbkSM1',
       'artist_name': 'Twenty One Pilots',
       'artist_uri': 'spotify:artist:3YQKmKGau1PzlVlkL1iodx',
       'duration_ms': 202333,
       'pos': 16,
       'track_name': 'Stressed Out',
       'track_uri': 'spotify:track:3CRDbSIZ4r5MsZ0YwxuEkn'},
      {'album_name': 'ANTI',
       'album_uri': 'spotify:album:2JNVoEx4psIgNQyEExwQVn',
       'artist_name': 'Rihanna',
       'artist_uri': 'spotify:artist:5pKCCKE2ajJHZ9KAiaK11H',
       'duration_ms': 219320,
       'pos': 17,
       'track_name': 'Work',
       'track_uri': 'spotify:track:14WWzenpaEgQZlqPq2nk4v'},
      {'album_name': 'x',
       'album_uri': 'spotify:album:6NoBzYmh5gUusGPCfg0pct',
       'artist_name': 'Rudimental',
       'artist_uri': 'spotify:artist:4WN5naL3ofxrVBgFpguzKo',
       'duration_ms': 242440,
       'pos': 18,
       'track_name': 'Lay It All On Me (feat. Ed Sheeran)',
       'track_uri': 'spotify:track:0vbbhcA6okLzvsy6WSTlLg'},
      {'album_name': 'Be Right There',
       'album_uri': 'spotify:album:6W3qGdySLUVV9OYW98UWZn',
       'artist_name': 'Diplo',
       'artist_uri': 'spotify:artist:5fMUXHkw8R8eOP2RNVYEZX',
       'duration_ms': 237051,
       'pos': 19,
       'track_name': 'Be Right There',
       'track_uri': 'spotify:track:1Wuwsq0BK4Abd7gRitOhXl'},
      {'album_name': 'Kamikaze',
       'album_uri': 'spotify:album:6L3ecTkK0aKTUIV7x0l1lU',
       'artist_name': 'MØ',
       'artist_uri': 'spotify:artist:0bdfiayQAKewqEvaU6rXCv',
       'duration_ms': 214240,
       'pos': 20,
       'track_name': 'Kamikaze',
       'track_uri': 'spotify:track:6TjP1C3sGQH1jBfzUXfcMg'},
      {'album_name': 'Ocean Drive',
       'album_uri': 'spotify:album:1jlUEbR1VEEGzjK47Xk1gT',
       'artist_name': 'Duke Dumont',
       'artist_uri': 'spotify:artist:61lyPtntblHJvA7FMMhi7E',
       'duration_ms': 206320,
       'pos': 21,
       'track_name': 'Ocean Drive',
       'track_uri': 'spotify:track:0k93MXOj0kSXo84SvSDeUz'},
      {'album_name': 'Hollow',
       'album_uri': 'spotify:album:50klDOXDGk4k5UneECuCwR',
       'artist_name': 'Tori Kelly',
       'artist_uri': 'spotify:artist:1vSN1fsvrzpbttOYGsliDr',
       'duration_ms': 210606,
       'pos': 22,
       'track_name': 'Hollow',
       'track_uri': 'spotify:track:160hN2OsDXnmgExwtG7cvD'},
      {'album_name': 'Revival',
       'album_uri': 'spotify:album:3Kbuu2tHsIbplFUkB7a5oE',
       'artist_name': 'Selena Gomez',
       'artist_uri': 'spotify:artist:0C8ZW7ezQVs4URX5aX7Kqx',
       'duration_ms': 200680,
       'pos': 23,
       'track_name': 'Hands To Myself',
       'track_uri': 'spotify:track:3CJvmtWw2bJsudbAC5uCQk'},
      {'album_name': 'Revival',
       'album_uri': 'spotify:album:7lDBDk8OQarV5dBMu3qrdz',
       'artist_name': 'Selena Gomez',
       'artist_uri': 'spotify:artist:0C8ZW7ezQVs4URX5aX7Kqx',
       'duration_ms': 207733,
       'pos': 24,
       'track_name': 'Body Heat',
       'track_uri': 'spotify:track:5VQ0SPGs7vdzQCIzsHTNUz'},
      {'album_name': 'Revival',
       'album_uri': 'spotify:album:3Kbuu2tHsIbplFUkB7a5oE',
       'artist_name': 'Selena Gomez',
       'artist_uri': 'spotify:artist:0C8ZW7ezQVs4URX5aX7Kqx',
       'duration_ms': 210506,
       'pos': 25,
       'track_name': 'Me & My Girls',
       'track_uri': 'spotify:track:4i55R0S2I64mhS5biNbECZ'}]}


