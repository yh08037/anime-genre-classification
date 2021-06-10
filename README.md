# anime-genre-classification
Predicting Anime Genres using NLP – Multi-Label Classification

## Dataset
### anime.csv
| MAL_ID | Name | Type | Genre | Synopsis |
|--------|------|------|-------|----------|
| ... | ... | ... | ... | ... |
| 9253 | Steins;Gate | TV | Thriller, Sci-Fi | The self-proclaimed mad scientist Rintarou Okabe rents out a room in a rickety old building in Akihabara, where he indulges himself in his hobby of inventing prospective "future gadgets" with fellow lab members: Mayuri Shiina, his air-headed childhood friend, and Hashida Itaru, a perverted hacker nicknamed "Daru." The three pass the time by tinkering with their most promising contraption yet, a machine dubbed the "Phone Microwave," which performs the strange function of morphing bananas into piles of green gel. Though miraculous in itself, the phenomenon doesn't provide anything concrete in Okabe's search for a scientific breakthrough; that is, until the lab members are spurred into action by a string of mysterious happenings before stumbling upon an unexpected success—the Phone Microwave can send emails to the past, altering the flow of history. Adapted from the critically acclaimed visual novel by 5pb. and Nitroplus, Steins;Gate takes Okabe through the depths of scientific theory and practicality. Forced across the diverging threads of past and present, Okabe must shoulder the burdens that come with holding the key to the realm of time. | 
| ... | ... | ... | ... | ... |

## Problem: Multi-Label Classification

|  | Example |
|---|---|
| Input:<br>  Synopsis | Koyomi Araragi, a third-year high school student, manages to survive a vampire attack with the help of Meme Oshino, a strange man residing in an abandoned building. Though being saved from vampirism and now a human again, several side effects such as superhuman healing abilities and enhanced vision still remain. Regardless, Araragi tries to live the life of a normal student, with the help of his friend and the class president, Tsubasa Hanekawa. When fellow classmate Hitagi Senjougahara falls down the stairs and is caught by Araragi, the boy realizes that the girl is unnaturally weightless. Despite Senjougahara's protests, Araragi insists he help her, deciding to enlist the aid of Oshino, the very man who had once helped him with his own predicament. Through several tales involving demons and gods, Bakemonogatari follows Araragi as he attempts to help those who suffer from supernatural maladies. |
| Ouput:<br>  Gernes | Romance, Supernatural, Mystery, Vampire |


## Original Dataset
- Kaggle link: [Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)<br>
- Github link: [MyAnimeList Database 2020](https://github.com/Hernan4444/MyAnimeList-Database)<br>

## Reference
[Movie genre prediction](https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/)
