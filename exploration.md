---
title: Exploration
---

<a name="top" />

## Motivation
Beyond the broader motivations for this area of study previous discussed in the overview, we formulated our approach based on our exploratory data analysis, our own experience using and testing Spotify’s platform, and the literature review described later on. 

In particular, we found that popularity was, as expected, a powerful song and playlist attribute measure and utilized it when evaluating the caliber of existing songs and playlists. Moreover, we noted that playlists are the sum of the songs within as well as synergies between these songs (rather than purely the sum of the parts). Of course, negative synergies are possible and something to be mindful of in evaluation and construction. Lastly, through our EDA, we better understand the core attributes of songs and playlists and were more informed on additional variables we could construct from the given data to measure things like sentiment and be able to enrich the data informing our model development. 
<br>

## Description of Data

Our main source of data is the Spotify API; our secondary data source was the Million Playlists Data. We explored other possible datasets, including the Last.fm (as discussed in Milestone 2) but found these to be insufficient in their scope and depth of coverage to fully capture the connections and details needed for an accurate and effective playlist generation model.   

The primary units of analysis in the data are playlists and tracks with the multiple tracks making up each playlist. Each playlist and track have descriptive attributes with which they are associated - these attributes are multi-faceted in their descriptive capabilities, underscoring, for example, the danceability, duration, energy, acousticness, loudiness, speechiness, liveness, key, etc. Furthermore, information on categories are associated with each playlist and track. Possible categories (can be thought of as something akin to “genre”) include pop, rock, indie/alternative, classical, jazz, punk, funk, world, family, toplist, etc. Each playlist and track is identified by a unique ID. Playlists can be matched to its member tracks. Additional functions can be utilized on the data to extract information on the engagement with these playlists and tracks, such as the getting details on the number of followers, the primary markets where the songs are available, and whether the song was featured or not. The majority of data, at the unit level, is either numeric or string; through proper extraction from arrays and dictionaries, the data has not been difficult to work with from the perspective of type compatibility.  

For initial explorations, we examined different visualizations and preliminary analyses of the most popular artists, tracks, and keywords. We created distribution plots to understand popularity and number of follower trends across playlists at the aggregate level. We also sought to better understand how different characteristics (such as degree of acousticness, valence, speechiness, tempo, etc.) of tracks or playlists mapped to popularity score and number of followers (as well as these two measures related to each other - indicating an overall positive relationship, as expected) to better understand how different types of songs appealed to people at the aggregate level. Further details on the results of our initial explorations are included below in the visualizations and findings section.

With regards to data cleaning,  we removed non existent playlists and track IDs ad removed the following track audio-features in a second version of our data set (analysis_url, track_href, uri, and type) as these are not relevant to our future model and broader goal of song discovery. We also removed duplicate columns. For data reconciliation, we compared the data saved in the file and the data before file saving (parameters) to ensure that the two are the same (to ensure the accuracy and preserve the integrity of the data before late use). 
<br>

## Literature Review

_He, Xiangnan; Liao, Lizi; Zhang, Hanwang; Nie, Liqiang; Hu, Xia; Chua, Tat-Seng. “Neural Collaborative Filtering.” National University of Singapore Press, 2018._ [Link to the paper.](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf)

Neural Collaborative Filtering (NCF) is a novel approach with the capabilities of deep learning applied to recommender systems design through collaborative filtering, the core interaction between users and items. NCF, unlike existing solutions, are able to generalize matrix factorization through replacing the inner product of the model with neural architecture. The paper finds that deeper layers of neural networks provide superior recommendation capabilities, and the overall NCF performs better than existing solutions. This paper informed our model construction, particularly since we first explored a neural network approach, as we discussed in our Milestone 3 submission, and subsequently a collaborative filtering approach so a synergistic combination of the 2 methods in a novel manner was incredibly useful. 
<br>

_Zamani, Hamed; Schedl, Markus; Lamere, Paul; Chen, Ching-Wei. “An Analysis of Approaches Taken in ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation.”_ [Link to the paper.](https://arxiv.org/abs/1810.01520) 

This paper analyzes the approaches and results of general performers and top performers in the RecSys Challenge. They find that matrix factorization, neighborhood based collaborative filtering models, and rank models were most commonly used across the board. They also find that models work best when enough tracks per a playlist are provided and are randomly selected, rather than sequentially selected. Interestingly, no submissions attempted to infer user intention, which stands in contrast to our original motivation mentioned earlier. The true differentiating results between models arises when many tracks per a playlist are used - when only a few tracks per a playlist are given, many models perform very similarly. Most submissions utilized the features directly given in the Spotify API; only a few teams attempted to formulate their own new data values from the raw audio directly. It is also interesting to note that in general, more information did not lead to superior recommendation capabilities - rather, it appears that more information restricted the generalizability of models and on net balanced out the potential added benefit of having additional data. 

Here, a [similar study](https://github.com/mrthlinh/Spotify-Playlist-Recommender?fbclid=IwAR04ojCadzSdXXXmzlhJMeWokPE8w3y8DBzz0mcMPfpfoFDfU3pqPhCpSVA) was completed to construct continuations of existing playlists given a set of features of the existing playlist. Most relevantly, the project utilized interesting metrics that we considered for our model design and evaluation. Specifically, they utilized the metric of R-precision - the number of retrieved relevant tracks divided by the number of known relevant tracks; they also utilized the normalized discounted cumulative gain from the R-precision - the ranking quality of the recommended tracks, increased where relevant tracks are placed relatively higher on the list. Their proposed solutions were similar to our initial design, focused on KNN, collaborative filtering, and matrix factorization as well as frequent pattern growth. The authors find that playlist based and song based KNN perform well on the dataset. Collaborative filtering achieves a similar result but is generally less efficient to implement.
<br>

_O’Bryant, Jacob. “A survey of music recommendation and possible improvements.” April 2017_. [Link to the paper.](https://pdfs.semanticscholar.org/7442/c1ebd6c9ceafa8979f683c5b1584d659b728.pdf)

This paper is a meta analysis of existing literature on music recommendations (not solely Spotify). In particular, they study collaborative and content based filtering and propose (without significant implementation) a combined approach that uses user skipping behavior to drive the model learning, balancing exploration and exploitation in generating the optimal listening experience for the user. While the exact practicalities of the design proposed are not made entirely clear and while the constraints of our data (for example, not having information historically or in real time data on user skipping behavior) limited our ability to construct or implement this model exactly, we did find it useful to get a survey of the landscape and different approaches, finding collaborative filtering to be the best fit for our data and interests. 
<br>

<a name="code" />

## Data retrieval code

* [Million Playlist Dataset](#mpd)
* [Spotify API Dataset](#spotAPI)

<a name="mpd" />

#### Million Playlist Dataset retrieval

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
        ...
        
[Return to code section selection.](#code)

[Return to top](#top)

<a name="spotAPI" />

#### Spotify API data retrieval

```python
import numpy as np
import pandas as pd
import sys
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import os
import pickle
import h5py
import json
```


```python
a = open("data/mpd.slice.0-999.json").read()
data = json.loads(a)
```


```python
track_URI=[0 for i in range(1000)]
artist_name=[0 for i in range(1000)]
track_name = [0 for i in range(1000)]
artist_URI = [0 for i  in range(1000)]
for i,song in enumerate(data["playlists"][0]["tracks"]):
    track_URI[i] = song["track_uri"]
    track_name[i] = '"' + song["track_name"] + '"'
    artist_name[i] = '"' + song["artist_name"] + '"'
    artist_URI[i] = song["artist_uri"]
```


```python
track_name[0]
```




    '"Lose Control (feat. Ciara & Fat Man Scoop)"'




```python
playlist_name = data["playlists"][1]["name"]
playlist_name = '"' + playlist_name + '"'
```


```python
#please enter your own information should you wish to run the file
username = 'xxx'
client_id ='xxx'
client_secret ='xxx'
scope = 'xxx'
redirect_uri = 'xxx'

token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
spotify = spotipy.Spotify(auth = token)
```


```python
results = spotify.search(q=track_name[0], type = 'track')
```


```python
results
```




    {'tracks': {'href': 'https://api.spotify.com/v1/search?query=%22Lose+Control+%28feat.+Ciara+%26+Fat+Man+Scoop%29%22&type=track&market=US&offset=0&limit=10',
      'items': [{'album': {'album_type': 'album',
         'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/2wIVse2owClT7go1WT98tk'},
           'href': 'https://api.spotify.com/v1/artists/2wIVse2owClT7go1WT98tk',
           'id': '2wIVse2owClT7go1WT98tk',
           'name': 'Missy Elliott',
           'type': 'artist',
           'uri': 'spotify:artist:2wIVse2owClT7go1WT98tk'}],
         'available_markets': ['AD', 'AR', ..., 'ZA'],
         'external_urls': {'spotify': 'https://open.spotify.com/album/6vV5UrXcfyQD1wu4Qo2I9K'},
         'href': 'https://api.spotify.com/v1/albums/6vV5UrXcfyQD1wu4Qo2I9K',
         'id': '6vV5UrXcfyQD1wu4Qo2I9K',
         ...
         'name': 'The Cookbook',
         'release_date': '2005-07-04',
         'release_date_precision': 'day',
         'total_tracks': 16,
         'type': 'album',
         'uri': 'spotify:album:6vV5UrXcfyQD1wu4Qo2I9K'},

```python
playlist_name
```




    '"Awesome Playlist"'




```python
artist_name[0]
```




    '"Missy Elliott"'




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
categories = sp.categories(country=None, locale=None, limit=50, offset=0)

#Get the ids of categories
temp = categories['categories']

cat_ids = []
for item in temp['items']:
    cat_ids.append(item['id'])
```


```python
playlists = {}
for cat_id in cat_ids:
    playlists[cat_id] = sp.category_playlists(category_id=cat_id, country=None, limit=50, offset=0)

playlist_ids_by_cat = {}
for category, playlist in playlists.items():
    #print(playlist['playlists']['items'][0]['id'])
    playlist_ids_by_cat[category] = [x['id'] for x in playlist['playlists']['items']]
```


```python
def spotify_id_to_isrc(spotify_ids):
    '''
    converts spotify ids to isrcs
    '''
    tracks = sp.tracks(spotify_ids)
    return [x['external_ids']['isrc']  for x in tracks['tracks']]

def isrc_to_spotify_id(isrcs):
    '''
    converts isrcs to spotify ids (not necessarily a 1 to 1 mapping)
    ''' 
    ids = []
    for isrc in isrcs:
        ids.append(sp.search('isrc:'+isrc)['tracks']['items'][0]['id'])
    return ids

def get_popularity_and_markets(spotify_ids):
    '''
    with a list of spotify IDs, get the popularity and market information for each song
    '''
    chunk_size= 42
    tmp = {}
    for i in range(0, len(spotify_ids), chunk_size):
        chunk = spotify_ids[i:i+chunk_size]
        features = sp.tracks(chunk)['tracks']
        features  = pd.DataFrame([x for x in features if isinstance(x, dict)])
        tmp_df = pd.DataFrame(features)
        tmp_df.index = tmp_df['id']
        tmp_df = tmp_df[['id', 'popularity', 'available_markets']]
        tmp_df['available_markets'] = tmp_df['available_markets'].apply(len)
        tmp.update(tmp_df.T.to_dict())
    df = pd.DataFrame(tmp).T
    return df

def get_followers(playlist_id, user = 'spotify'):
    '''
    find the number of follower of a playlist
    '''
    playlist = sp.user_playlist(user, playlist_id=playlist_id, fields = ['followers'])
    return playlist['followers']['total']

#Default: US, 11/24/2017 at 8PM
def get_featured_playlists(country = 'US', time = '2017-11-24T18:00:00'):
    '''
    determine whether playlist was featured on a given date and time
    '''
    featured = sp.featured_playlists(country=country, timestamp=time, limit=50, offset=0)
    return [x['id'] for x in featured['playlists']['items']]

def get_track_ids(playlist_id = '37i9dQZF1DX3FNkD0kDpDV'):
    ''' 
    give spotify IDs for the songs in a playlist
    '''
    offset = 0
    playlist = sp.user_playlist_tracks(user = 'spotify', playlist_id = playlist_id, limit = 100)
    ids = [x['track']['id']  for x in playlist['items']]
    # if we hit the limit, need to add more
    while len(ids) / (offset + 100) == 1:
        offset = offset + 100
        playlist = sp.user_playlist_tracks(user = 'spotify', playlist_id = playlist_id, limit = 100, offset = offset)
        ids = ids + [x['track']['id']  for x in playlist['items']]
    return ids

def get_track_audio_features(spotify_ids = get_track_ids()):
    'given a list of spotify IDs, return a dataframe of track audio features'
    chunk_size= 42
    tmp = {}
    for i in range(0, len(spotify_ids), chunk_size):
        chunk = spotify_ids[i:i+chunk_size]
        features = sp.audio_features(chunk)
        if features:
            tmp_df = pd.DataFrame([x for x in features if isinstance(x, dict)])
            tmp_df.index = tmp_df['id']
            tmp.update(tmp_df.T.to_dict())
    df = pd.DataFrame(tmp).T
    df = df.drop(['analysis_url', 'track_href', 'uri', 'type'], 1)
    return df
```


```python
import time

def get_playlist_data(playlist_ids):
    '''
    Given a list of Spotify playlist IDs, returns a dataframe containing a row
    for each inputed playlist with columns for the following data:
    a) *average* audio characteristics for the songs in that playlist:
        acousticness, danceability, duration,
        energy, instrumentalness, key, liveness, loudness, mode, tempo,
        valence, and time signature
    b) average popularity of songs in the playlist
    c) popularity of most popular song in playlist - determines whether and which song drives the playlist essentially
    d) average number of markets the songs in the playlist are available in
    e) global playlist info
        - number of followers of playlist
        - number of tracks in playlist
        - whether or not the playlist was "featured" on certain date at certain time
    '''
    rez = {}
    # force list
    if not isinstance(playlist_ids, list):
        playlist_ids = [playlist_ids]
        
    featured_playlists = get_featured_playlists()
    audio_char_dict = {}
    popularity_dict = {}
    for playlist_id in playlist_ids:
        print('Getting info for: ' + playlist_id)
        tmp = {}
        try:
            track_ids = get_track_ids(playlist_id)
        except spotipy.client.SpotifyException:
            print('WARNING: Playlist does not exist. Skipping.')
            continue
        except TypeError:
            print("id is empty")
            continue
        except:
            time.sleep(10)
            track_ids = get_track_ids(playlist_id)
        # get average audio characteristics
        audio_chars = get_track_audio_features(track_ids)
        audio_char_mean = audio_chars.mean().to_dict()
        # get popularity and markets
        pop_and_mkts = get_popularity_and_markets(track_ids)
        pop_and_mkts_mean = pop_and_mkts.mean().to_dict()
        audio_char_dict.update(audio_chars.T.to_dict())
        popularity_dict.update(pop_and_mkts.T.to_dict())
        # get # followers
        tmp['num_followers'] = get_followers(playlist_id)
        tmp['num_tracks'] = len(track_ids)
        tmp['featured'] = 1 if playlist_id in featured_playlists else 0
        tmp.update(audio_char_mean)
        tmp.update(pop_and_mkts_mean)
        rez[playlist_id] = tmp
    return pd.DataFrame(rez).T, popularity_dict, audio_char_dict


# Loop through all of our playlists and get popularity info, track feature info, and 
progress = 0
playlist_level_data = {}
song_level_data_pop = {}
song_level_data_audio = {}
for cat, playlists in playlist_ids_by_cat.items():
    if playlists[0] in playlist_level_data.keys():
        continue
    sp = refresh()
    #print('Starting Category: ' + cat)
    playlist_data, pop, audio = get_playlist_data(playlists)
    playlist_data['category'] = cat
    song_level_data_pop.update(pop)
    song_level_data_audio.update(audio)
    playlist_level_data.update(playlist_data.T.to_dict())
```

    Getting info for: 37i9dQZF1DXcBWIGoYBM5M
    Getting info for: 37i9dQZF1DX0XUsuxWHRQd
    Getting info for: 37i9dQZF1DXcF6B6QPhFDv
    ...
    id does not exist
    Getting info for: 37i9dQZF1DX6BsbcWKm1XO

```python
'''
loop through playlists and store the track IDs from each playlists;
we use this to get a mapping of playlists to tracks
'''
playlist_to_tracks = {}
progress = 1
for cat, playlists in playlist_ids_by_cat.items():
    print(progress / len(playlist_ids_by_cat))
    for playlist in playlists:
        try:
            track_ids = get_track_ids(playlist)
        except spotipy.client.SpotifyException:
            print('WARNING: Playlist does not exist. Skipping.')
            continue
        except TypeError:
            print("id is empty")
            continue
        except:
            time.sleep(10)
            track_ids = get_track_ids(playlist)
        playlist_to_tracks[playlist] = track_ids
    progress += 1
```

    0.024390243902439025
    0.04878048780487805
    id is empty
    0.07317073170731707
    0.0975609756097561
    ...
    0.8780487804878049
    0.9024390243902439
    0.926829268292683
    0.9512195121951219
    0.975609756097561
    1.0
    


```python
playlist_to_tracks
```




    {'37i9dQZF1DXcBWIGoYBM5M': ['2rPE9A1vEgShuZxxzR2tZH',
      '5p7ujcrUXASCNwRaWNHR1C',
      '2LskIZrCeLxRvCiGP8gxlh',
      ...
      '1A6OTy97kk0mMdm78rHsm8',
      ...
      '30I4xIABs2mAQHznGlB9fz']],
     ...}




```python
playlist_features = {}
for playlist in playlist_to_tracks:
    playlist_features[playlist] = get_track_audio_features(playlist_to_tracks[playlist])
```

    C:\Users\Elbert\Anaconda3\lib\site-packages\ipykernel_launcher.py:77: UserWarning: DataFrame columns are not unique, some columns will be omitted.
    


```python
playlist_features
```

[Return to code section selection.](#code)

[Return to top](#top)
