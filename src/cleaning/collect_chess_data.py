import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

chess_titles = ['GM', 'WGM','IM','WIM', 'FM', 'WFM', 'NM', 'WNM', 'CM', 'WCM']

def get_titled_players_ids():
    for title in chess_titles:
        url = 'https://api.chess.com/pub/titled/'+title
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json',
        }
        try:
            response = requests.get(url,headers=headers)
            response.raise_for_status()
            data = response.json()

            open(title+'_IDs.txt', 'w').write('\n'.join(data['players']))

        except requests.exceptions.HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')
        except requests.exceptions.RequestException as err:
            print(f'Other error occurred: {err}')
        except ValueError:
            print('Response content is not valid JSON.')


def get_players_and_ratings():
    for title in tqdm(chess_titles):
        with open(title+'_IDs.txt', 'r') as file:
            ids = file.read().splitlines()
            for player_id in tqdm(ids):
                url = 'https://api.chess.com/pub/player/'+player_id+'/stats'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                    'Accept': 'application/json',
                }
                try:
                    response = requests.get(url,headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    if 'chess_blitz' in data and 'best' in data['chess_blitz'] and 'last' in data['chess_blitz']:
                        last_blitz_rating = str(data['chess_blitz']['last']['rating'])
                        best_blitz_rating = str(data['chess_blitz']['best']['rating'])
                        open('blitz_ratings.txt', 'a').write(', '.join([player_id,title, last_blitz_rating,best_blitz_rating])+'\n')

                    if 'chess_bullet' in data and 'best' in data['chess_bullet'] and 'last' in data['chess_bullet']:
                        last_bullet_rating = str(data['chess_bullet']['last']['rating'])
                        best_bullet_rating = str(data['chess_bullet']['best']['rating'])
                        open('bullet_ratings.txt', 'a').write(', '.join([player_id,title, last_bullet_rating,best_bullet_rating])+'\n')

                except requests.exceptions.HTTPError as http_err:
                    print(f'HTTP error occurred: {http_err}')
                except requests.exceptions.RequestException as err:
                    print(f'Other error occurred: {err}')
                except ValueError:
                    print('Response content is not valid JSON.')

def sort_players():
    for csv_file in ['bullet_ratings.csv', 'blitz_ratings.csv']:
        df = pd.read_csv(csv_file)
        breakpoint()
        sort_column = 'username'
        df_sorted = df.sort_values(by=sort_column)

        df_sorted.to_csv(csv_file, index=False)

def plot_ratings():
    for csv_file in ['bullet_ratings.csv', 'blitz_ratings.csv']:
        df = pd.read_csv(csv_file)
        print(f"type(df['last rating']):{type(df['last rating'])}")
        #df['last rating'].plot(kind='hist', bins=10, edgecolor='black', title='Distribution of Chess Ratings')
        x = np.sort(df['last rating'].values)
        y = np.arange(1, len(x)+1) / len(x)

        plt.plot(x, y, marker='.', linestyle='-')
        plt.xlabel('Value')
        plt.ylabel('CDF')
        plt.title('Empirical CDF (smooth line)')
        plt.grid(True)
        plt.show()

if __name__=="__main__":
    get_titled_players_ids()
    get_players_and_ratings()

    # sort_players()
    # plot_ratings()
