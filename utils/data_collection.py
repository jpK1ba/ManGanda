#!/usr/bin/env python
# coding: utf-8


import re, shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm, trange
from utils.pickling import *
from torchvision import transforms


def get_attrs(url):
    """Get titles, urls, and ratings in the rankings page"""
    soup = bs4.BeautifulSoup(requests.get(url).content)
    mangas = soup.select('tr[class="ranking-list"]')
    
    # Titles
    titles = [
        manga.select_one('h3[class="manga_h3"]').text for manga in mangas
    ]
    
    # URLs
    urls = [
        manga.select_one('h3[class="manga_h3"] a').get('href') 
        for manga in mangas
    ]
    
    # Ratings
    ratings = [
        float(manga.select_one('td[class="score ac fs14"]').text.strip())
        for manga in mangas
    ]
    
    # Check Results
    if not len(titles) == len(urls) == len(ratings):
        raise Error
    
    return titles, urls, ratings


def get_details(url):
    """Get n_faves, genres, and authors in the manga page"""
    soup = bs4.BeautifulSoup(requests.get(url).content)
    
    # Number of Favorites
    try:
        n_fav = [list(n_fav.parent.children)[1].text
                 for n_fav in soup.select('div[class="spaceit_pad"] span')
                 if n_fav.text == 'Favorites:'][0]
        n_fav = int(re.sub(r'\D', '', n_fav))
    except:
        n_fav = None
    
    # Genre
    try:
        genre = [genre.text for genre in 
                 soup.select('div[class="spaceit_pad"]'
                             ' span[itemprop="genre"]')
                 if genre.parent.select_one('span').text == 'Genres:']
    except:
        genre = []
    
    # Authors
    try:
        author = [author.text for author in 
                  soup.select('div[class="spaceit_pad"] a')
                  if author.parent.select_one('span').text == 'Authors:']
    except:
        author = []
        
    return n_fav, genre, author


def get_data(limit):
    """Crawl `myanimelist.net` for the list of mangas and details"""
    # Load data if available
    try:
        df = load_pkl(f'df_{limit}')
        
    except:
        # Instatiate Features
        titles  = []
        urls    = []
        ratings = []

        # Crawl Ranking Pages (https://myanimelist.net/topmanga.php?limit=0)
        for start in trange(0,limit,50):
            url = f'https://myanimelist.net/topmanga.php?limit={start}'
            try:
                results = get_attrs(url)
            except:
                print(f'Crawling Ended Prematurely, Stopped @ {start}')
                break

            # Update
            titles.extend(results[0])
            urls.extend(results[1])
            ratings.extend(results[2])

        # Instatiate Other Details
        n_favs  = []
        genres  = []
        authors = []

        # Scrape each Manga Page (https://myanimelist.net/manga/2/Berserk)
        for url in tqdm(urls):
            details = get_details(url)

            # Update
            n_favs.append(details[0])
            genres.append(details[1])
            authors.append(details[2])

        df = pd.DataFrame({
            'titles': titles,
            'urls': urls,
            'ratings': ratings,
            'n_favs': n_favs,
            'genres': genres,
            'authors': authors,
        })
        save_pkl(df, f'df_{limit}')
        df.to_csv(f'df_{limit}.csv', index=False)
                 
    return df


def download_samples(title, n=3):
    """Download manga pages"""
    # Format title
    title_url = re.sub(r'\W', '_', title.title())
    url = f'https://w15.mangafreak.net/Read1_{title_url}_1'
    soup = bs4.BeautifulSoup(requests.get(url, timeout=2).content)
    
    # Check if manga is found
    home_title = 'Read Free Manga Online - MangaFreak'
    if soup.select_one('title').text == home_title:
        return False # Skip if not
        
    else:
        # Check for folder
        if not os.path.exists(f'./data/{title_url.lower()}'):
            os.mkdir(f'./data/{title_url.lower()}')
        
        # Check for contents
        if len(os.listdir(f'./data/{title_url.lower()}')) < 5*n:

            # Get Chapters
            chapters = [re.findall(r'_(\d+)', chapter.get('value'))[0]
                        for chapter in soup.select('option')
                        if 'Read1' in chapter.get('value')]

            # Remove folder if no valid chapters
            if len(set(chapters)) < 5:
                shutil.rmtree(f'./data/{title_url.lower()}',
                              ignore_errors=True)
                return False

            # Get Random Chapters
            for chapter in np.random.choice(chapters, 5, replace=False):
                c_url = (f'https://w15.mangafreak.net/'
                         f'Read1_{title_url}_{chapter}')
                chap_soup = bs4.BeautifulSoup(
                    requests.get(c_url, timeout=2).content
                )

                # Get Number of Pages
                try:
                    pages = int(re.findall(r'(\d+) pages',
                                           chap_soup.text)[0])
                except:
                    try:
                        pages = int([re.findall(r'Page (\d+)',
                                                page.get('alt'))[0]
                                     for page in chap_soup.select('img')
                                     if 'Page' in page.get('alt')][-1])
                    except:
                        pages = 10

                title_lower = title_url.lower()

                # Download images
                half = pages // 2

                for page in range(half-int(n/2), half+round(n/2)):
                    url = (f'https://images.mangafreak.net/mangas'
                           f'/{title_lower}/{title_lower}_{chapter}'
                           f'/{title_lower}_{chapter}_{page}.jpg')
                    get_ipython().system("wget -q '{url}' -P './data/{title_lower}'")
    
    # If all is successful
    return True


def download_mangas(df, n=3):
    """Download images samples for the listed mangas"""
    # Create folder for data
    if not os.path.exists('./data'):
        os.mkdir('./data')
    
        # Get list of mangas with downloadable images
        mangas = []

        # For each manga
        for title in tqdm(df['titles'].tolist()):

            try:
                # Download samples
                if download_samples(title, n=3):
                    # Append the title to mangas if filled with images
                    mangas.append(title)
            except:
                continue

        # Get the relevant mangas
        df['check'] = (
            df['titles'].apply(lambda x: re.sub(r'\W', '_', x.lower()))
        )
        df.set_index('titles', inplace=True)

        df_manga = df.loc[mangas]

        print('Downloading Samples: Success!')
        
    else:
        mangas = os.listdir('./data')
        
        df['check'] = (
            df['titles'].apply(lambda x: re.sub(r'\W', '_', x.lower()))
        )
        df = df.sort_values('ratings').drop_duplicates('titles', keep='last')
        df.set_index('titles', inplace=True)
        df_manga = df[df['check'].isin(mangas)]
        
        print('Samples are already downloaded!')
    
    print(f'Download Rate: {(len(df_manga)/len(df)*100):.2f}%')
    return df_manga


def get_annotations(df_data):
    """Get annotations for dataset"""
    if os.path.exists('saves/annotations.csv'):
        pass
    else:
        df_data = df_data.reset_index().set_index('check')

        data_dir = './data'
        mangas = os.listdir(data_dir)


        df_metadata = pd.DataFrame()
        for manga in tqdm(mangas):
            if manga[0] == '.':
                continue
            manga_path = os.path.join(data_dir, manga)
            paths = [os.path.join(manga_path, x)
                     for x in os.listdir(manga_path)]

            # Check data
            verified = []
            for path in paths:
                try:
                    transforms.ToTensor()(Image.open(path))
                    verified.append(path)
                except:
                    continue

            # Save good data
            df_manga = pd.DataFrame({'paths': verified})
            df_manga['rating'] = [df_data.loc[manga, 'ratings']]*len(verified)
            df_manga['title'] = [df_data.loc[manga, 'titles']]*len(verified)

            df_metadata = pd.concat([df_metadata, df_manga])

        df_metadata.to_csv('saves/annotations.csv', index=False)