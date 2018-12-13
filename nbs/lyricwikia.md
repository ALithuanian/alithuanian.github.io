

```python
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
ccm = SpotifyClientCredentials("X", "Y")
sp = spotipy.Spotify(client_credentials_manager=ccm)
```

### Lyrics wikia data


```python
# pack songs into packs of 50 for faster API queries
songs = song_audio_df
song_packs = []
for i in range(0,len(songs),50):
    if i+50 < len(songs):
        song_packs.append(songs["id"][i:i+50])
    else:
        song_packs.append(songs["id"][i:])
```


```python
# find all song titles and artist names
import time
all_songs = []
for pack in song_packs:
    response = sp.tracks(pack)["tracks"]
    for song in response:
        track, artist = song["name"], song["artists"][0]["name"]
#         print(track,artist)
        all_songs.append([track,artist])
    time.sleep(0.10) 
```


```python
to_save = pd.DataFrame(all_songs, columns=["song","artist"])
to_save = to_save[to_save.song != ""]
to_save.to_csv("titles_artists.csv", sep="|")
```


```python
songs = pd.read_csv("titles_artists.csv", sep="|", encoding='latin1', index_col=0)
```


```python
songs["songl"] = songs.song.map(lambda x: re.search("[^;()-]+", x).group())
songs.songl = songs.songl.str.lower().str.rstrip()
songs["artistl"] = songs.artist.map(lambda x: re.search("[^;()-]+", x).group())
songs.artistl = songs.artistl.str.lower().str.rstrip()
```

#### Using Lyricwikia API


```python
# https://pypi.org/project/lyricwikia/
import lyricwikia
def f(row):
    try:
        l = lyricwikia.get_lyrics(row.artistl, row.songl).replace("\n\n","\n")
    except:
        l = "na"
    return l if l != "na" else 0
```


```python
songs["lyrics"] = songs.apply(f, axis=1)
```


```python
len(songs[songs.lyrics != 0]) / len(songs)
```

Using Lyricwikia API we could only obtain lyrics for ~27.3% of songs.

#### Finding missing data by scraping manually


```python
def search_missing_lyrics(df):
    homepage = "http://lyrics.wikia.com"
    search_url = homepage + "/wiki/Special:Search?query="
    
    # search for artist
    search = requests.get(search_url+df.artistl)
    search_soup = bs(search.content, "lxml")
    artist_found = search_soup.find("li", class_="result")
    if not artist_found:
        return 0
    artist_link = artist_found.find("a")["href"]
    if not artist_link:
        return 0
    
    # search for song per artist
    artist = requests.get(artist_link)
    artist_soup = bs(artist.content, "lxml")
    ### CHANGE SONG
    song_title = re.compile("^{}$".format(df.songl), re.I)
    song_found = artist_soup.find("a", text=song_title)
    if not song_found:
#         print("song not found")
        return 0
    song_link = homepage + song_found["href"]
    
    # get song lyrics
    song = requests.get(song_link)
    song_soup = bs(song.content, "lxml")
    lyrics = song_soup.find("div", class_="lyricbox")
    if not lyrics:
#         print("lyrics not found")
        return 0
    for br in lyrics.find_all("br"):
        br.replace_with(" ")
    
    return lyrics.text
```


```python
songs["lyrics"] = songs.apply(search_missing_lyrics, axis=1)
```


```python
songs["lyrics"] = songs.lyrics.lower()
songs["lyrics"] = re.sub(r"[\?,!;:-]","", songs.lyrics)
songs["lyrics"] = re.sub(r" ?\([^)]+\)", "", songs.lyrics)
```


```python
len(songs[songs.lyrics != 0]) / len(songs)
```

While we could find lyrics for more songs after scraping, the percentage is still low at ~36.7%. 

Let's look at the most frequently used words in songs 


```python
all_words = pd.Series(songs.lyrics.str.cat(sep=' ').split())
all_words.value_counts().hist()
```

<img src="eda/lyrwik.png" align="left" alt="drawing" width="500"/>

It seems that using Lyricwikia data for our playlist enhancer is not an option in this case because of 1) low lyrics coverage from the largest free lyric data source; and 2) high frequency of standard words that are useless for differentiation.
