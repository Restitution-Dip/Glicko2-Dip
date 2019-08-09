from urllib import request
from Config import GR_DATA_URL, PICKLE_FOLDER, USE_CACHED_GAMEDATA
import os
import pandas
import numpy


def load_gr_data():

    with request.urlopen(GR_DATA_URL) as f:
        new_etag = f.info()['ETag']

    if not os.path.exists('gr_data.csv'):
        request.urlretrieve(GR_DATA_URL, filename='gr_data.csv')

        if os.path.exists('etag'):
            os.remove('etag')

        f = open('etag', 'w+')
        f.write(new_etag)
        f.close()

    else:
        if not os.path.exists('etag'):
            print('no etag exists: redownloading the file.')
            os.remove('gr_data.csv')
            request.urlretrieve(GR_DATA_URL, filename='gr_data.csv')
            print('file retrieved')
            f = open('etag', 'w+')
            f.write(new_etag)
            f.close()
            print('etag written')
        else:
            f = open('etag', 'r')
            old_etag = f.read()

            if old_etag == new_etag:
                print('no download necessary: etag is same and file exists.')

            else:
                os.remove('gr_data.csv')
                os.remove('etag')
                request.urlretrieve(GR_DATA_URL, filename='gr_data.csv')
                f = open('etag', 'w+')
                f.write(new_etag)
                f.close()

    new_etag = new_etag.replace("\"", "")
    etag_pickle_folder = os.path.join(PICKLE_FOLDER, new_etag)

    if not os.path.exists(etag_pickle_folder):
        os.mkdir(etag_pickle_folder)

    assert os.path.exists(etag_pickle_folder)

    game_data = pandas.read_csv('gr_data.csv', encoding='ANSI', low_memory=False)

    return game_data, etag_pickle_folder

def get_history_and_user_data(game_data, etag_pickle_folder):
    history_data_path = os.path.join(etag_pickle_folder, 'history_data.pkl')
    user_data_path = os.path.join(etag_pickle_folder, 'user_data.pkl')
    if not os.path.exists(history_data_path) or not os.path.exists(
            user_data_path) or not USE_CACHED_GAMEDATA:
        history_data = game_data.dropna()
        history_data['variantID'] = pandas.to_numeric(history_data['variantID'])
        history_data['userID'] = pandas.to_numeric(history_data['userID'])
        history_data['processTime'] = pandas.to_numeric(history_data['processTime'])

        hist_data_orig_length = len(history_data)
        history_data = history_data[(history_data.potType == 'Sum-of-squares') |
                                    (history_data.potType == 'Winner-takes-all')]

        history_data.loc[:, 'points'] = 0
        history_data.loc[:, 'points'] = numpy.where((history_data['status'] == 'Drawn') &
                                                    (history_data['potType'] == 'Sum-of-squares'),
                                                    history_data['supplyCenterNo'] *
                                                    history_data['supplyCenterNo'],
                                                    history_data['points'])
        history_data.loc[:, 'points'] = numpy.where((history_data['status'] == 'Drawn') &
                                                    (history_data['potType'] == 'Winner-takes-all'),
                                                    1,
                                                    history_data['points'])
        history_data.loc[:, 'points'] = numpy.where(history_data['status'] == 'Won', 1,
                                                    history_data['points'])
        points_sum = history_data.groupby('gameID').agg({'points': 'sum'})
        points_sum.columns = ['points_sum']
        history_data = history_data.join(points_sum, on='gameID')
        history_data['points_share'] = history_data['points'] / history_data['points_sum']
        history_data.to_pickle(history_data_path)

        user_data = game_data[hist_data_orig_length:]
        assert len(user_data) + hist_data_orig_length == len(game_data)
        user_data = user_data[user_data.columns[0:3]]
        user_data = user_data[1:]
        user_data.columns = ['userID', 'username', 'is_banned']
        user_data['userID'] = pandas.to_numeric(user_data['userID'])
        user_data['is_banned'] = pandas.to_numeric(user_data['is_banned'], errors='coerce',
                                                   downcast='integer')

        user_data.set_index('userID', inplace=True, drop=True)
        user_data.to_pickle(user_data_path)

    user_data = pandas.read_pickle(user_data_path)
    history_data = pandas.read_pickle(history_data_path)

    return history_data, user_data
