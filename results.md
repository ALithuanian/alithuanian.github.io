---
title: Outcomes
---

<a name="top" />

* [Results](#results)
* [Conclusion and Summary](#conclsum)
* [Future Work](#futwork)

<a name="results" />

## Results

None of our models were able to accurately predict which songs were in any given playlist. However, we believe that the collaborative filtering approach with a more comprehensive dataset would have been able to do such a thing. When we ran this model on a set of slightly over 300 playlists, 250 of which were labeled “latin,” and 114 of which labeled “classical,” we discovered that despite sharing categories, these playlists still have abysmally low similarities to one another. As a result, we would see at most 2 songs of overlap between playlists, and very few songs not in the truncated playlist were suggested with any significant probability. We did notice that the probabilities for the songs in the test set did tend to be at least 1%, though, which indicated that there was some playlist with nonzero overlap which contained the song.

When we tested this model on a dataset consisting of 2000 playlists pulled from the million playlists dataset, we noticed that the MSE dropped from about 0.95 to about 0.85; that is, for the values in the test set that should have been 1, if the model predicted a song to be in the playlist with probability p, the average of (1-p)^2 dropped from about 0.95 to about 0.85. This was a fairly significant drop, and we believe that if we were to increase the data further so we could have a substantial number of similar playlists to any given playlist, this model would see even greater success. Unfortunately, we did not have the computation power necessary to test this.

We also found that through the Million Playlist Data, there were a number of playlists with just 1 follower (which is quite typical as expected from user experience where we create a playlist just for ourselves or ourselves and one friend, rather than some large group or some organically mass followed playlist). Consequently, the values in our similarity matrix were quite low, which led to the aforementioned results found in evaluation. One possible way to ameliorate this is through taking a top k approach, which would reduce the focus and incorporation of these 1 follower playlists that otherwise confound the data and thereby model inputs and model. More broadly, as we mentioned in our literature review, approaches that account for natural bias of this sort and others through the UI understanding we noted here, are quite valuable in constructing improved models.

[Return to top](#top)
<br>

<a name="conclsum" />

## Conclusion and Summary
	
As discussed in the overview and motivation sections, through this project, we aimed to better understand the performance of different models in generating playlists across different types of listeners and music categories. We also aimed to generate a novel method, building on existing work, that generates playlists for automated song discovery. We focused primarily on considerations of accuracy of song discovery and secondarily, considerations around context (including UI inferences - for example, what would users truly care about when evaluating new songs or other recommendation songs?) as well as practical implementation and execution advantages and disadvantages.  

We were inspired to pursue this work by the growth in centralized, platform based music streaming services like Spotify and felt that there was tremendous potential in work on recommender systems in this particular area given the rapid user base growth on music streaming platforms and Spotify in particular as well as the aforementioned uniqueness of song discovery as a challenging but rewarding area of recommender system application.

[Return to top](#top)
<br>

<a name="futwork" />

## Future Work

There are several possible avenues of future studies to build upon our work here. In particular, we can explore: 

1. using a top k approach, as mentioned above, and general accounting for bias to provide better data inputs and likely improve model performance per our aforementioned understanding of the current raw data we have retrieved
2. how the prediction model capabilities across other possible segmentation of song, playlist, and user types
3. the efficacy of content based vs. collaborative filtering approaches
4. the usage of different evaluation metrics in determining the caliber of prediction models
5. incorporating user interaction for more direct user real time feedback in learning based models
6. the effect of inclusion and exclusion of different types of data, such as intention, mood, and geography
7. working with different data sets (such as focusing more on the Million Playlist Data and/or incorporating LyricWiki and the crowdsourced user given tags in the Last.fm data)

[Return to top](#top)
