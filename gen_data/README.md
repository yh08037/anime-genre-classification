## 1) get_html.ipynb
Jupyter Notebook to download and unzip html files scrapped from MyAnimeList by [Hernan Valdivieso](https://github.com/Hernan4444).

Since the [original data on kaggle](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020?select=anime_with_synopsis.csv)
contains few missing letters in synopsis column, extracting the data again from the original html.

This notebook requires `curl` to download html data from [Hernan Valdivieso](https://github.com/Hernan4444)'s Google Drive.

After execution, a new folder `html` will be created.

```
anime-genre-classification/
├─ clean_data/
├─ gen_data/
│  ├─ html/
|  │  ├─ 1.html
|  │  ├─ 5.html
|  │  ├─ ...
│  ├─ generate_data.ipynb
│  ├─ get_html.ipynb
├─ ptb_dataset/
│  ├─ ptb.py
```


## 2) generate_data.ipynb
Jupyter Notebook to generate csv file from html data created by `get_html.ipynb`.

This notebook requires [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/) to parse html string.

After execution, a new file `anime.csv` will be created, and the folder `html` will be deleted.

```
anime-genre-classification/
├─ clean_data/
├─ gen_data/
│  ├─ anime.csv
│  ├─ generate_data.ipynb
│  ├─ get_html.ipynb
├─ ptb_dataset/
│  ├─ ptb.py
```
