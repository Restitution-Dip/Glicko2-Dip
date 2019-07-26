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

_MAXSIZE = 186
_MAXMULTI = .272
_MULTISLOPE = .00391
_WIN = 1.0
_LOSS = 0
_CATCH = .5
_INITRAT = 1500
_INITCONF = 35.0

_TYPICAL_CONF = 15
_CONF_INC_PER_MONTH = math.sqrt((_INITCONF ** 2 - _TYPICAL_CONF ** 2) / 100)
_INITVOL = .005
_VOL = .2
_CONV = 173.7178
_EPS = 0.001
# TODO: order by timestamp
start = time.time()
gr_data_url = 'http://www.webdiplomacy.net/ghostRatingData.txt'
USE_POINT_DIFFERENCE = True
chunks = []
tqdm.pandas()
import time

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
                oppPhi = j.confidence / _CONV

                if USE_POINT_DIFFERENCE:
                    S = ((i.pot_share - j.pot_share) / 2) + .5
                else:
                    if i.pot_share == 0 and j.pot_share == 0:
                        S = 0.5
                    else:
                        S = (i.pot_share / (i.pot_share + j.pot_share))

                # Change the weight of the matchup based on opponent confidence
                weighted = 1 / math.sqrt(1 + 3 * (oppPhi ** 2) / (math.pi ** 2))

                # Change the weight of the matchup based on competition size
                # weighted = weighted * multi
                expected_score = 1 / (1 + math.exp(-weighted * (mu - oppMu)))
                # expected_score = 1 / (1 + math.exp(-1 * (mu - oppMu)))
                v_inv += weighted ** 2 * expected_score * (1 - expected_score)
                delta += weighted * (S - expected_score)
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
    def __init__(self, userID, pot_share, rating, confidence, volatility):
        self.userID = userID
        self.pot_share = pot_share
        self.rating = rating
        self.confidence = confidence
        self.volatility = volatility

    def __eq__(self, other):
        return self.userID == other.userID

    def __hash__(self):
        return hash(self.userID)


class RatingPeriod:
    competitors = []

    def addCompetitor(self, userID, pot_share, rating, confidence,
                      volatility):
        competitor = GlickoPlayer(userID, pot_share, rating, confidence, volatility)
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
            meet.addCompetitor(userID, points_share, rating, confidence, vol)
        else:
            # Initial ratings if a player hasn't competed before
            meet.addCompetitor(userID, points_share, _INITRAT, _INITCONF, _INITVOL)
    if len(meet.competitors) > 1:
        calculate_change_over_time(meet.competitors, entries, process_time)
        calculateGlicko(meet.competitors)

        # Take results of competition and append data

        for player in meet.competitors:
            ratings[player.userID] = [player.rating, player.confidence, player.volatility]
            if player.userID in entries:
                res_list = entries.get(player.userID)
                res_list.append([game_id,
                                 player.rating,
                                 player.confidence,
                                 process_time])
                entries[player.userID] = res_list
            else:
                entries[player.userID] = [[game_id,
                                           player.rating,
                                           player.confidence,
                                           process_time]]


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

algo_start_time = time.time()
game_data = pandas.read_csv('gr_data.csv', encoding='ANSI', low_memory=False)
history_data = game_data.dropna()

user_data = game_data[len(history_data):]
assert len(user_data) + len(history_data) == len(game_data)
user_data = user_data[user_data.columns[0:3]]
user_data = user_data[1:]
user_data.columns = ['userID', 'username', 'is_banned']

user_data['userID'] = pandas.to_numeric(user_data['userID'])
user_data['is_banned'] = pandas.to_numeric(user_data['is_banned'], errors='coerce',
                                           downcast='integer')

user_data.set_index('userID', inplace=True, drop=True)

history_data['variantID'] = pandas.to_numeric(history_data['variantID'])
history_data['userID'] = pandas.to_numeric(history_data['userID'])
history_data['processTime'] = pandas.to_numeric(history_data['processTime'])

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

most_recent_timestamp = history_data['processTime'].max()
most_recent_time = datetime.fromtimestamp(most_recent_timestamp)
print('Most recent game was on {}'.format(most_recent_time))
now = time.time()
now = datetime.fromtimestamp(now)
print('This script is being run on {}'.format(now))
print('It has been {} since the data source was updated.'.format(now - most_recent_time))
points_sum = history_data.groupby('gameID').agg({'points': 'sum'})
points_sum.columns = ['points_sum']
history_data = history_data.join(points_sum, on='gameID')
history_data['points_share'] = history_data['points'] / history_data['points_sum']

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
classic_gb_sos = classic_gb.loc[(classic_gb['potType'] == 'Sum-of-squares')]
classic_gb_dss = classic_gb.loc[(classic_gb['potType'] == 'Winner-takes-all')]


def get_ratings_and_entries_for_slice(slice, activity_filter=90, filter_inactive=True,
                                      filter_banned=True, placement_games=10):
    games = slice.sort_values(by=['processTime'])
    games = games.groupby('gameID', sort=False)
    slice_ratings = {}
    slice_entries = {}

    for game_id, game in games:
        game = game.loc[:, ['gameID', 'userID', 'points_share', 'processTime']]
        do_glicko(game, slice_ratings, slice_entries)

    ratings = pandas.DataFrame.from_dict(slice_ratings, orient='index',
                                         columns=['rating', 'confidence', 'volatility'])
    old_ratings_len = len(ratings)
    ratings = ratings.join(user_data, how='inner')
    assert old_ratings_len == len(ratings)
    if filter_inactive:
        time_limit = algo_start_time - (86400 * activity_filter)
        active_users = slice[slice['processTime'] >= time_limit]['userID'].drop_duplicates()

        ratings = ratings.loc[active_users]
    if filter_banned:
        ratings = ratings[ratings['is_banned'] == 0]
    if placement_games >= 0:
        user_ids = slice['userID']
        num_games = user_ids.value_counts(sort=True)
        num_games = num_games[num_games >= placement_games]
        ratings = ratings[ratings.index.isin(num_games.index)]

    ratings['rating_lowerbound'] = ratings['rating'] - ratings['confidence']
    ratings = ratings.sort_values('rating_lowerbound', ascending=False)
    rank_lb = ratings['rating_lowerbound'].rank(ascending=False)
    rank_pure = ratings['rating'].rank(ascending=False)

    ratings['rank_lb'] = rank_lb
    ratings['rank_pure'] = rank_pure
    ratings.index.name = 'userID'

    def expected_score_vs_player(ratings, opp_rating, opp_confidence):
        opp_mu = (opp_rating - _INITRAT) / _CONV
        opp_phi = confidence_med / _CONV
        opp_weight = 1 / math.sqrt(1 + 3 * (opp_phi ** 2) / (math.pi ** 2))
        all_mu_diff = -opp_weight * (((ratings['rating'] - _INITRAT) / _CONV) - opp_mu)
        all_mu_diff = all_mu_diff.apply(math.exp)
        expected_score_vs_opp = 1 / (1 + all_mu_diff)
        expected_score_vs_opp = (0.5 - expected_score_vs_opp) * 2
        return expected_score_vs_opp

    rating_med = ratings['rating'].median()
    confidence_med = ratings['confidence'].median()
    ratings['expected_score_vs_median'] = expected_score_vs_player(ratings, rating_med,
                                                                   confidence_med)

    ratings_mean = ratings['rating'].mean()
    confidence_mean = ratings['confidence'].mean()
    ratings['expected_score_vs_mean'] = expected_score_vs_player(ratings, ratings_mean,
                                                                 confidence_mean)

    best_rating = ratings.iloc[0]['rating']
    best_confidence = ratings.iloc[0]['confidence']
    ratings['expected_score_vs_best'] = expected_score_vs_player(ratings, best_rating,
                                                                 best_confidence)

    ratings = ratings[
        ['username', 'rating_lowerbound', 'rating', 'confidence', 'expected_score_vs_median',
         'expected_score_vs_mean', 'expected_score_vs_best', 'rank_lb', 'rank_pure']]
    return ratings, slice_entries


cfg_ratings, cfg_entries = get_ratings_and_entries_for_slice(classic_fullpress, 60, True,
                                                             placement_games=5)

# classic_full_sos_live_ratings, classic_full_sos_live_entries = get_ratings_and_entries_for_slice(
#     classic_fullpress_sos_live, placement_games=0, activity_filter=365, filter_inactive=False)
classic_full_sos_nl_ratings, classic_full_sos_nl_entries = get_ratings_and_entries_for_slice(
    classic_fullpress_sos_nl, placement_games=5)

# classic_full_dss_live_ratings, classic_full_dss_live_entries = get_ratings_and_entries_for_slice(
#     classic_fullpress_dss_live, placement_games=5)
classic_full_dss_nl_ratings, classic_full_dss_nl_entries = get_ratings_and_entries_for_slice(
    classic_fullpress_dss_nl, placement_games=5)

classic_full_live_ratings, classic_full_live_entries = get_ratings_and_entries_for_slice(
    classic_fullpress_live, placement_games=5)

gb_ratings, gb_entries = get_ratings_and_entries_for_slice(classic_gb, placement_games=5)
gb_sos_ratings, gb_sos_entries = get_ratings_and_entries_for_slice(classic_gb_sos,
                                                                   placement_games=5)
gb_dss_ratings, gb_dss_entries = get_ratings_and_entries_for_slice(classic_gb_dss,
                                                                   placement_games=5)

chaos_ratings, chaos_entries = get_ratings_and_entries_for_slice(chaos, placement_games=0)

important_players = ['jmo1121109', 'Restitution', 'bo_sox48', 'Squigs44', 'Carl Tuckerson']
print("Classic FP top-20:")
print(tabulate(cfg_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = cfg_ratings[cfg_ratings['username'].isin(important_players)]
if len(important) > 0: print(tabulate(important, showindex=True, headers="keys"))

print("Classic FP (DSS) (Non-Live) top-20:")
print(tabulate(classic_full_dss_nl_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = classic_full_dss_nl_ratings[
    classic_full_dss_nl_ratings['username'].isin(important_players)]
if len(important) > 0: print(tabulate(important, showindex=True, headers="keys"))

print("Classic FP (SoS) (Non-Live) top-20:")
print(tabulate(classic_full_sos_nl_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = classic_full_sos_nl_ratings[
    classic_full_sos_nl_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

# print("Classic FP (DSS) (Live) top-20:")
# print(tabulate(classic_full_dss_live_ratings.head(20), showindex=True, headers="keys"))
#
# print("Classic FP (SoS) (Live) top-20:")
# print(tabulate(classic_full_sos_live_ratings.head(20), showindex=True, headers="keys"))

print("Classic FP (Live) top-20:")
print(tabulate(classic_full_live_ratings))
print("Important players in previous category:")
important = classic_full_live_ratings[classic_full_live_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

print("Classic GB top-20:")
print(tabulate(gb_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = gb_ratings[gb_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

print("Classic GB (DSS) top-20:")
print(tabulate(gb_dss_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = gb_dss_ratings[gb_dss_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

print("Classic GB (SoS) top-20")
print(tabulate(gb_sos_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = gb_sos_ratings[gb_sos_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

print("Chaos top-20:")
print(tabulate(chaos_ratings.head(20), showindex=True, headers="keys"))
print("Important players in previous category:")
important = chaos_ratings[chaos_ratings['username'].isin(important_players)]
if len(important) > 0:
    print(tabulate(important, showindex=True, headers="keys"))

end = time.time()

print("Time to process all: {}".format(end - start))

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
