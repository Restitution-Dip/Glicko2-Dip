import pandas
import numpy
import hashlib
from tqdm import tqdm
from urllib import request
import pickle
import os
import math
import time
from tabulate import tabulate
from datetime import datetime
import statistics
PICKLE_FOLDER = 'pickles'

USE_PARTIAL_WINS = True

_MAXSIZE = 186
_MAXMULTI = .272
_MULTISLOPE = .00391
_WIN = 1.0
_LOSS = 0
_CATCH = .5
_INITRAT = 1500
_INITCONF = 100

_TYPICAL_CONF = 15
_CONF_INC_PER_MONTH = math.sqrt((_INITCONF ** 2 - _TYPICAL_CONF ** 2) / 100)
_INITVOL = .005
_VOL = .1
_CONV = 173.7178
_EPS = 0.001
# TODO: order by timestamp
gr_data_url = 'http://www.webdiplomacy.net/ghostRatingData.txt'
USE_POINT_DIFFERENCE = True
chunks = []
tqdm.pandas()
import time
start = time.time()

USE_CACHED_RATINGS = True
USE_CACHED_GAMEDATA = True
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)


def calculate_change_over_time(players, entries, process_time):
    for player in players:
        if player.userID in entries:
            time_since_last_game = process_time - entries[player.userID][-1][3]
            months_since_last_game = time_since_last_game / 2.592e+6
            if months_since_last_game > 1:
                new_confidence = min(_INITCONF,
                                     math.sqrt(
                                         player.confidence ** 2 +
                                         _CONF_INC_PER_MONTH ** 2 * months_since_last_game))
                player.confidence = new_confidence


def calculateGlicko(players):
    N = len(players)
    if N > _MAXSIZE:
        multi = _MAXMULTI
    else:
        multi = _WIN - _MULTISLOPE * N
    # compare every head to head matchup in a given compeition
    for i in players:
        mu = (i.rating - _INITRAT) / _CONV
        phi = i.confidence / _CONV
        sigma = i.volatility
        v_inv = 0
        delta = 0
        for j in players:
            if i is not j:
                oppMu = (j.rating - _INITRAT) / _CONV
                i.rating_diffs.append(i.rating - j.rating)
                oppPhi = j.confidence / _CONV

                # Change the weight of the matchup based on opponent confidence
                weighted = 1 / math.sqrt(1 + 3 * (oppPhi ** 2) / (math.pi ** 2))
                # Change the weight of the matchup based on competition size
                weighted = weighted * multi
                expected_score = 1 / (1 + math.exp(-weighted * (mu - oppMu)))

                if USE_PARTIAL_WINS:
                    if USE_POINT_DIFFERENCE:
                        S = ((i.pot_share - j.pot_share) / 2) + .5
                    else:
                        if i.pot_share == 0 and j.pot_share == 0:
                            S = 0.5
                        else:
                            S = (i.pot_share / (i.pot_share + j.pot_share))

                    # expected_score = 1 / (1 + math.exp(-1 * (mu - oppMu)))
                    v_inv += weighted ** 2 * expected_score * (1 - expected_score)
                    delta += weighted * (S - expected_score)
                else:
                    diff = i.pot_share - j.pot_share
                    if diff > 0:
                        S = 1.0
                    elif diff == 0:
                        S = 0.5
                    else:
                        S = 0.0

                    # Weight adjusted by point difference
                    weighted = weighted * abs(diff)
                    v_inv += weighted ** 2 * expected_score * (1 - expected_score)
                    delta += (weighted * (S - expected_score))
                i.scores.append(i.pot_share - j.pot_share)


        if v_inv != 0:
            v = 1 / v_inv
            change = v * delta
            newSigma = findSigma(mu, phi, sigma, change, v)
            phiAst = math.sqrt(phi ** 2 + newSigma ** 2)
            # New confidence based on competitors volatility and v
            newPhi = 1 / math.sqrt(1 / phiAst ** 2 + 1 / v)

            newMu = mu + newPhi ** 2 * delta
            i.rating = newMu * _CONV + _INITRAT
            i.confidence = newPhi * _CONV
            i.volatility = newSigma


def get_expected_score(p1_rating, p2_rating, p2_confidence):
    mu = (p1_rating - _INITRAT) / _CONV
    oppMu = (p2_rating - _INITRAT) / _CONV
    oppPhi = p2_confidence / _CONV

    weighted = 1 / math.sqrt(1 + 3 * (oppPhi ** 2) / (math.pi ** 2))
    # weighted = 1
    expected_score = 1 / (1 + math.exp(-weighted * (mu - oppMu)))
    if USE_POINT_DIFFERENCE:
        expected_score = expected_score - 0.5
        expected_score = expected_score * 2
    return expected_score


def get_expected_score_by_username(ratings, userid_1, userid_2):
    ratings_1 = ratings.loc[ratings['username'] == userid_1]
    ratings_2 = ratings.loc[ratings['username'] == userid_2]
    mu = (ratings_1['rating'].iloc[0] - _INITRAT) / _CONV
    oppMu = (ratings_2['rating'].iloc[0] - _INITRAT) / _CONV

    return get_expected_score(ratings_1['rating'].iloc[0],
                              ratings_2['rating'].iloc[0],
                              ratings_2['confidence'].iloc[0])


def findSigma(mu, phi, sigma, change, v):
    alpha = math.log(sigma ** 2)

    def f(x):
        tmp = phi ** 2 + v + math.exp(x)
        a = math.exp(x) * (change ** 2 - tmp) / (2 * tmp ** 2)
        b = (x - alpha) / (_VOL ** 2)
        return a - b

    a = alpha
    if change ** 2 > phi ** 2 + v:
        b = math.log(change ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(alpha - k * _VOL) < 0:
            k += 1
        b = alpha - k * _VOL
    fa = f(a)
    fb = f(b)
    # Larger _EPS used to speed iterations up slightly
    while abs(b - a) > _EPS:
        c = a + (a - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb < 0:
            a = b
            fa = fb
        else:
            fa /= 2
        b = c
        fb = fc
    return math.e ** (a / 2)


class GlickoPlayer:
    def __init__(self, userID, pot_share, rating, confidence, volatility, scores):
        self.userID = userID
        self.pot_share = pot_share
        self.rating = rating
        self.confidence = confidence
        self.volatility = volatility
        self.rating_diffs = []
        self.scores = scores

    def __eq__(self, other):
        return self.userID == other.userID

    def __hash__(self):
        return hash(self.userID)


class RatingPeriod:
    competitors = []

    def addCompetitor(self, userID, pot_share, rating, confidence,
                      volatility, scores):
        competitor = GlickoPlayer(userID, pot_share, rating, confidence, volatility, scores)
        self.competitors.append(competitor)


def do_glicko(game, ratings, entries):
    # Add players to competition and calculate ratings

    meet = RatingPeriod()
    meet.competitors = []
    game_id = None
    process_time = None
    for idx, dat in game.iterrows():
        game_id = dat['gameID']
        userID = dat['userID']
        points_share = dat['points_share']
        process_time = dat['processTime']

        if userID in ratings:
            rating = float(ratings.get(userID)[0])
            confidence = float(ratings.get(userID)[1])
            vol = float(ratings.get(userID)[2])
            scores = ratings.get(userID)[4]
            meet.addCompetitor(userID, points_share, rating, confidence, vol, scores)
        else:
            # Initial ratings if a player hasn't competed before
            meet.addCompetitor(userID, points_share, _INITRAT, _INITCONF, _INITVOL, [])
    if len(meet.competitors) > 1:
        calculate_change_over_time(meet.competitors, entries, process_time)
        calculateGlicko(meet.competitors)

        # Take results of competition and append data

        for player in meet.competitors:
            ratings[player.userID] = [player.rating, player.confidence, player.volatility, player.rating_diffs, player.scores]
            if player.userID in entries:
                res_list = entries.get(player.userID)
                res_list.append([game_id,
                                 player.rating,
                                 player.confidence,
                                 process_time,
                                 player.pot_share])
                entries[player.userID] = res_list
            else:
                entries[player.userID] = [[game_id,
                                           player.rating,
                                           player.confidence,
                                           process_time,
                                           player.pot_share]]

with request.urlopen(gr_data_url) as f:
    new_etag = f.info()['ETag']

if not os.path.exists('gr_data.csv'):
    request.urlretrieve(gr_data_url, filename='gr_data.csv')

    if os.path.exists('etag'):
        os.remove('etag')

    f = open('etag', 'w+')
    f.write(new_etag)
    f.close()

else:
    if not os.path.exists('etag'):
        print('no etag exists: redownloading the file.')
        os.remove('gr_data.csv')
        request.urlretrieve(gr_data_url, filename='gr_data.csv')
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
            request.urlretrieve(gr_data_url, filename='gr_data.csv')
            f = open('etag', 'w+')
            f.write(new_etag)
            f.close()

new_etag = new_etag.replace("\"", "")
etag_pickle_folder = os.path.join(PICKLE_FOLDER, new_etag)

if not os.path.exists(etag_pickle_folder):
    os.mkdir(etag_pickle_folder)

assert os.path.exists(etag_pickle_folder)

algo_start_time = time.time()
game_data = pandas.read_csv('gr_data.csv', encoding='ANSI', low_memory=False)

history_data_path = os.path.join(etag_pickle_folder, 'history_data.pkl')
user_data_path = os.path.join(etag_pickle_folder, 'user_data.pkl')
if not os.path.exists(history_data_path) or not os.path.exists(user_data_path) or not USE_CACHED_GAMEDATA:
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

most_recent_timestamp = history_data['processTime'].max()
most_recent_time = datetime.fromtimestamp(most_recent_timestamp)
print('Most recent game was on {}'.format(most_recent_time))
now = time.time()
now = datetime.fromtimestamp(now)
print('This script is being run on {}'.format(now))
print('It has been {} since the data source was updated.'.format(now - most_recent_time))

chaos = history_data.loc[(history_data['variantID'] == 17)]
classic = history_data.loc[(history_data['variantID'] == 1)]

classic_fullpress = classic.loc[(classic['pressType'] == 'Regular')]

classic_fullpress_live = classic_fullpress[classic_fullpress['phaseMinutes'] <= 60]
# classic_fullpress_sos_live = classic_fullpress_live.loc[(classic_fullpress_live['potType'] == 'Sum-of-squares')]
# classic_fullpress_dss_live = classic_fullpress_live.loc[(classic_fullpress_live['potType'] == 'Winner-takes-all')]

classic_fullpress_nl = classic_fullpress[classic_fullpress['phaseMinutes'] > 60]
classic_fullpress_sos_nl = classic_fullpress_nl.loc[
    (classic_fullpress_nl['potType'] == 'Sum-of-squares')]
classic_fullpress_dss_nl = classic_fullpress_nl.loc[
    (classic_fullpress_nl['potType'] == 'Winner-takes-all')]

classic_gb = classic.loc[(classic['pressType'] == 'NoPress')]
classic_gb_nl = classic_gb[classic_gb['phaseMinutes'] > 60]
classic_gb_live = classic_gb[classic_gb['phaseMinutes'] <= 60]
classic_gb_sos_nl = classic_gb_nl.loc[(classic_gb['potType'] == 'Sum-of-squares')]
classic_gb_dss_nl = classic_gb_nl.loc[(classic_gb['potType'] == 'Winner-takes-all')]

def get_ratings_and_entries_for_slice(slice, activity_filter=90, filter_inactive=True,
                                      filter_banned=True, placement_games=10):
    games = slice.sort_values(by=['processTime'])
    games = games.groupby('gameID', sort=False)
    slice_ratings = {}
    slice_entries = {}

    for game_id, game in games:
        game = game.loc[:, ['gameID', 'userID', 'points_share', 'processTime']]
        do_glicko(game, slice_ratings, slice_entries)

    num_players_in_variant = len(game)
    ratings = pandas.DataFrame.from_dict(slice_ratings, orient='index',
                                         columns=['rating', 'confidence', 'volatility', 'rating_diffs', 'scores'])
    old_ratings_len = len(ratings)
    ratings = ratings.join(user_data, how='inner')
    assert old_ratings_len == len(ratings)
    if filter_inactive:
        time_limit = algo_start_time - (86400 * activity_filter)
        active_users = slice[slice['processTime'] >= time_limit]['userID'].drop_duplicates()

        ratings = ratings.loc[active_users]
    if filter_banned:
        ratings = ratings[ratings['is_banned'] == 0]
    user_ids = slice['userID']
    num_games = user_ids.value_counts(sort=True)
    if placement_games >= 0:
        num_games = num_games[num_games >= placement_games]
        ratings = ratings[ratings.index.isin(num_games.index)]
    ratings['num_games'] = num_games

    ratings['rating_lowerbound'] = ratings['rating'] - ratings['confidence']
    ratings = ratings.sort_values('rating_lowerbound', ascending=False)
    rank_lb = ratings['rating_lowerbound'].rank(ascending=False)
    rank_pure = ratings['rating'].rank(ascending=False)

    ratings['rank_lb'] = rank_lb
    ratings['rank_pure'] = rank_pure
    ratings.index.name = 'userID'

    ratings['rating_diffs'] = ratings['rating_diffs'].apply(lambda x: statistics.mean(x))
    ratings['scores'] = ratings['scores'].apply(lambda x: statistics.mean(x))
    def expected_score_vs_player(ratings, opp_rating, opp_confidence):
        opp_mu = (opp_rating - _INITRAT) / _CONV
        opp_phi = opp_confidence / _CONV
        opp_weight = 1 / math.sqrt(1 + 3 * (opp_phi ** 2) / (math.pi ** 2))
        all_mu_diff = -opp_weight * (((ratings['rating'] - _INITRAT) / _CONV) - opp_mu)
        all_mu_diff = all_mu_diff.apply(math.exp)
        expected_score_vs_opp = 1 / (1 + all_mu_diff)
        expected_score_vs_opp = (expected_score_vs_opp - 0.5) * 2
        return expected_score_vs_opp

    rating_med = ratings['rating'].median()
    confidence_med = ratings['confidence'].median()
    ratings['expected_score_vs_median'] = expected_score_vs_player(ratings, rating_med,
                                                                   confidence_med)

    ratings_mean = ratings['rating'].mean()
    confidence_mean = ratings['confidence'].mean()
    ratings['expected_score_vs_mean'] = expected_score_vs_player(ratings, ratings_mean,
                                                                 confidence_mean)

    ratings['expected_score_vs_new'] = expected_score_vs_player(ratings, _INITRAT, _INITCONF)

    best_rating = ratings.iloc[0]['rating']
    best_confidence = ratings.iloc[0]['confidence']
    ratings['expected_score_vs_best'] = expected_score_vs_player(ratings, best_rating,
                                                                 best_confidence)

    # ((1 / N) * (7-N)) / 6 = x, solve for N
    ratings['expected_draw_size_vs_mean'] = num_players_in_variant / ((ratings['expected_score_vs_mean'] * (num_players_in_variant - 1)) + 1)
    ratings['expected_score_in_mean_game'] = 1 / ratings['expected_draw_size_vs_mean']

    ratings = ratings[
        ['username', 'num_games', 'rating_lowerbound', 'rating', 'confidence', 'expected_score_vs_median',
         'expected_score_vs_mean', 'expected_score_vs_best', 'expected_score_vs_new', 'expected_score_in_mean_game', 'rating_diffs', 'rank_lb', 'rank_pure']]
    ratings.columns = ['name', 'num_games', 'rating_lb', 'rating', 'RD', 'vs_med',
         'vs_mean', 'vs_best', 'vs_new', 'ppg_vs_mean', 'avg_rating_diff', 'rank_lb', 'rank_pure']

    ratings.set_index('name', drop=True, inplace=True)
    return ratings, slice_entries


def _load_from_pickle(name, input_slice, activity_filter=90, filter_inactive=True, filter_banned=True,
                      placement_games=5):
    ratings_path = os.path.join(etag_pickle_folder, name + '.pkl')
    entries_path = os.path.join(etag_pickle_folder, name + '_entries.pkl')
    if not os.path.exists(ratings_path) or not os.path.exists(entries_path) or not USE_CACHED_RATINGS:
        ratings, entries = get_ratings_and_entries_for_slice(input_slice,
                                                             activity_filter=activity_filter,
                                                             filter_inactive=filter_inactive,
                                                             filter_banned=filter_banned,
                                                             placement_games=placement_games)
        with open(ratings_path, 'wb') as f:
            pickle.dump(ratings, f)
        with open(entries_path, 'wb') as f:
            pickle.dump(entries, f)

    with open(ratings_path, 'rb') as f:
        ratings = pickle.load(f)
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)
    return ratings, entries


classic_fullpress_ratings = _load_from_pickle('classic_fullpress_ratings',
                                              classic_fullpress,
                                              activity_filter=60,
                                              placement_games=10)
classic_fullpress_nl_ratings = _load_from_pickle('classic_fullpress_nl_ratings',
                                                 classic_fullpress_nl)
classic_full_sos_nl_ratings = _load_from_pickle('classic_fullpress_sos_nl_ratings',
                                                classic_fullpress_sos_nl)

classic_full_dss_nl_ratings = _load_from_pickle('classic_fullpress_dss__nl_ratings',
                                                classic_fullpress_dss_nl)

classic_full_live_ratings = _load_from_pickle('classic_full_live_ratings',
                                              classic_fullpress_live)

gb_ratings = _load_from_pickle('classic_gb_ratings',
                               classic_gb)
gb_sos_ratings_nl = _load_from_pickle('classic_gb_sos_nl_ratings',
                                   classic_gb_sos_nl)
gb_dss_ratings_nl = _load_from_pickle('classic_gb_dss_nl_ratings',
                                   classic_gb_dss_nl)
classic_gb_live_ratings = _load_from_pickle('classic_gb_live_ratings',
                                            classic_gb_live)
chaos_ratings = _load_from_pickle('chaos_ratings',
                                  chaos,
                                  placement_games=0)
IMPORTANT_PLAYERS = ['jmo1121109', 'Restitution', 'bo_sox48', 'Squigs44', 'Carl Tuckerson']

def tabulate_ratings(name, ratings, head=20):
    entries = ratings[1]
    ratings = ratings[0]
    print("{} top-{}:".format(name, head))
    print(tabulate(ratings.head(head), showindex=True, headers="keys"))
    important = ratings[ratings.index.isin(IMPORTANT_PLAYERS)]
    if len(important) > 0:
        print("{} important players:".format(name))
        print(tabulate(important, showindex=True, headers="keys"))
    print("\n")

classic_fullpress_nl_entries = classic_fullpress_nl_ratings[1]
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings[0]

gr_july_classic_fp_nl = pandas.read_csv('GhostRatings-2019-07 - FP-NL-CLA.csv')
gr_july_classic_fp_nl.set_index('Player', drop=True, inplace=True)
gr_july_classic_fp_nl = gr_july_classic_fp_nl[['Rank']]
gr_july_classic_fp_nl = gr_july_classic_fp_nl[gr_july_classic_fp_nl.index.isin(classic_fullpress_nl_ratings.index.values)]

gr_july_classic_fp_nl['GR_rank'] = gr_july_classic_fp_nl.rank(ascending=True)

len_before = len(classic_fullpress_nl_ratings)
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.join(gr_july_classic_fp_nl['GR_rank'], how='left')
assert len(classic_fullpress_nl_ratings) == len_before

classic_fullpress_nl_ratings['rank_diff'] = classic_fullpress_nl_ratings['GR_rank'] - classic_fullpress_nl_ratings['rank_lb']

tabulate_ratings("Classic FP", classic_fullpress_ratings)
tabulate_ratings("Classic FP (DSS) (Non-Live)", classic_full_dss_nl_ratings)
tabulate_ratings("Classic FP (SoS) (Non-Live)", classic_full_sos_nl_ratings)
tabulate_ratings("Classic FP (Live)", classic_full_live_ratings)
tabulate_ratings("Classic GB", gb_ratings)
tabulate_ratings("Classic GB (DSS) (Non-Live)", gb_dss_ratings_nl)
tabulate_ratings("Classic GB (SoS), (Non-Live)", gb_sos_ratings_nl)
tabulate_ratings("Classic GB (Live)", classic_gb_live_ratings)
tabulate_ratings("Classic Chaos", chaos_ratings)

print("Special tabulations:")
tabulate_ratings("Classic FP (Non-Live)", (classic_fullpress_nl_ratings, classic_fullpress_nl_entries))

classic_fullpress_nl_ratings['best_rank'] = classic_fullpress_nl_ratings['GR_rank'].combine(classic_fullpress_nl_ratings['rank_lb'], min, 0)
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings[classic_fullpress_nl_ratings['best_rank'] <= 30]
classic_fullpress_nl_ratings['abs_rank_diff'] = classic_fullpress_nl_ratings['rank_diff'].abs()
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['abs_rank_diff'], ascending=False)
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.head(100)
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['rank_diff'], ascending=False)
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['rank_lb'], ascending=True)
classic_fullpress_nl_ratings = classic_fullpress_nl_ratings[['rank_diff', 'num_games', 'avg_rating_diff', 'rank_lb', 'GR_rank']]
tabulate_ratings("Classic FP (Non-Live) Big Upsets", (classic_fullpress_nl_ratings, classic_fullpress_nl_entries), head=1000)

for name in classic_fullpress_nl_ratings.index.values:
    userID = user_data[user_data['username'] == name].index.values[0]
    list_of_games = classic_fullpress_nl_entries[userID]
    list_of_games = [(x[0], x[1], x[2], x[3], x[4]) for x in list_of_games]
    list_of_diffs = []
    prev_rat = 1500
    for game in list_of_games:
        rat = game[1]
        p_share = game[4]
        RD = game[2]
        list_of_diffs.append((rat - prev_rat, p_share, RD))
        prev_rat = rat

    print(name, list_of_diffs)

end = time.time()


print("Time to process all: {} seconds".format(end - start))

#
# cfg_ratings, cfg_entries = get_ratings_and_entries_for_slice(classic_fulblahpress, 60, True)
#
# print("Classic fullpress top-20:")
# print(cfg_ratings.head(20))


# ratings = user_data.join(ratings, how='right', on='userID')
# print(ratings)
# events = align_data(argv[1])15^
# count = 0
# for event in events:
#     if len(event) == 4:
#         print count
#         count += 1
#         name = smart_str(event[1][0])
#         date = event[0]
#         gender = event[2]
#         do_glicko(event[3], name, date, gender)
# sorted_boys = sorted(ratings_boys.items(), key=itemgetter(1))
# sorted_girls = sorted(ratings_girls.items(), key=itemgetter(1))
# write_rating(sorted_boys, "male")
# write_rating(sorted_girls, "female")
# write_ath(entries_girls)
# write_ath(entries_boys)
