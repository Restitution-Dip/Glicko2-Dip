from Config import *
import inspect
import hashlib
import os
import pickle
from tabulate import tabulate
import pandas
from Glicko import GlickoCalculator
import time
import statistics
import numpy
from bokeh.plotting import figure, output_file, show
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter

class CategoryResults:
    def __init__(self, name, input_slice, user_data, activity_filter=90, filter_inactive=True,
                 filter_banned=True, placement_games=5, glicko_calculator=GlickoCalculator()):
        arg_vals = inspect.getargvalues(inspect.currentframe())
        locals = arg_vals.locals
        locals['self'] = None
        locals['glicko_calculator'] = sorted(glicko_calculator.__dict__.items(), key=lambda x: x[0])
        sorted_locals = sorted(locals.items(), key=lambda x: x[0])
        checksum = hashlib.sha1(str(sorted_locals).encode('utf-8')).hexdigest()[-5:]

        from Config import ETAG_PICKLE_FOLDER
        if ETAG_PICKLE_FOLDER is None:
            raise Exception("ETAG_PICKLE_FOLDER was not set.")
        filename = name + '_' + checksum + '.pkl'
        file_path = os.path.join(ETAG_PICKLE_FOLDER, filename)
        if os.path.exists(file_path) and USE_CACHED_RATINGS:
            with open(file_path, 'rb') as f:
                self.__dict__.update(pickle.load(f))
        else:
            print("Generating: {}".format(filename))
            self.glicko_calculator = glicko_calculator
            self.user_data = user_data
            self.name = name
            ratings, entries = self.get_ratings_and_entries_for_slice(input_slice,
                                                                      activity_filter=activity_filter,
                                                                      filter_inactive=filter_inactive,
                                                                      filter_banned=filter_banned,
                                                                      placement_games=placement_games)
            self.ratings = ratings
            self.entries = entries

            with open(file_path, 'wb') as f:
                pickle.dump(self.__dict__, f)

    def tabulate(self, head=20):
        print("{} top-{}:".format(self.name, head))
        ratings_head = self.ratings.head(head)
        print(tabulate(ratings_head, showindex=True, headers="keys"))

        best_player = ratings_head.index.values[0]

        imp_players = IMPORTANT_PLAYERS.copy()
        imp_players.append(best_player)

        important = self.entries[self.entries.index.isin(imp_players, level=0)]
        if len(important) > 0:
            print("{} important players histories:".format(self.name))
            print(tabulate(important, showindex=True, headers="keys"))
        print("\n")

    def show_history(self, head=1):
        imp_players = list(self.ratings.head(head).index.values)
        imp_players.extend(IMPORTANT_PLAYERS)
        imp_players = set(imp_players)
        p = figure(plot_width=800, plot_height=400, x_axis_type='datetime')
        for imp_player in imp_players:
            imp_player_entries = self.entries[
                self.entries.index.get_level_values(0) == imp_player].reset_index()
            imp_player_entries['datetime'] = pandas.to_datetime(imp_player_entries['timestamp'],
                                                                unit='s')
            source = ColumnDataSource(imp_player_entries)

            p.circle(x='datetime', y='rating', source=source)
            p.step(x='datetime', y='rating', source=source, mode="after")
            p.add_tools(HoverTool(
                tooltips=[
                    ("gameid", "@gameid"),
                    ("Point share", "@point_share"),
                    ("RD change", "@RD_diff"),
                    ("Rating change", "@rating_diff")
                ]
            ))
            # p.step(date_times.values, imp_player_entries['rating'], mode="after")
            # p.circle(date_times.values, imp_player_entries['rating'])
        return p

    def get_history_plot(self, username):

        p = figure(plot_width=1620, plot_height=400, x_axis_type='datetime', title=self.name,
                   active_scroll='wheel_zoom')
        player_entries = self.entries[
            self.entries.index.get_level_values(0) == username].reset_index()
        player_entries['datetime'] = pandas.to_datetime(player_entries['timestamp'], unit='s')

        source = ColumnDataSource(player_entries)

        p.circle(x='datetime', y='rating', source=source)
        p.step(x='datetime', y='rating', source=source, mode="after")
        p.add_tools(HoverTool(
            tooltips=[
                ("gameid", "@gameid"),
                ("Point share", "@point_share"),
                ("RD change", "@RD_diff"),
                ("Rating change", "@rating_diff")
            ]
        ))
        return p

    def get_leaderboard(self):
        source = ColumnDataSource(self.ratings)
        float_formatter = NumberFormatter(format='0,0.00')
        percentage_formatter = NumberFormatter(format='0.0%')
        columns = []
        for col_name in self.ratings.columns:
            col_title = col_name
            formatter = float_formatter
            if col_name == 'num_games':
                col_title = '# Games'
            elif col_name == 'rating_lb':
                col_title = 'Rating (Lowerbound)'
            elif col_name == 'rating':
                col_title = 'Rating'
            elif 'vs_' in col_name:
                continue
            elif col_name == 'vs_med':
                col_title = "Expected point diff vs median"
                formatter = percentage_formatter
            elif col_name == 'vs_mean':
                col_title = "Expected point diff vs mean"
                formatter = percentage_formatter
            elif col_name == 'vs_best':
                col_title = "Expected point diff vs best"
                formatter = percentage_formatter
            elif col_name == 'ppg_vs_mean':
                col_title = 'Expected points per game vs mean'
                formatter = percentage_formatter
            elif col_name == 'avg_rating_diff':
                col_title = 'Average ratings advantage'
            elif col_name == 'rank_lb':
                col_title = 'Rank (Lowerbound)'
            elif col_name == 'rank_pure':
                col_title = 'Rank (Pure)'
            col_type = self.ratings[col_name].dtype

            if str(col_type) == 'int64':
                column = TableColumn(field=col_name, title=col_title)

            elif str(col_type) == 'float64':
                column = TableColumn(field=col_name, title=col_title, formatter=formatter)

            else:
                raise Exception("Unknown column type!")

            columns.append(column)
        columns = [TableColumn(field='name', title='Username')] + columns
        data_table = DataTable(columns=columns, index_position=None, source=source, width=1620)
        return data_table

    def get_ratings_and_entries_for_slice(self, slice, activity_filter, filter_inactive,
                                          filter_banned, placement_games):
        games = slice.sort_values(by=['processTime'])
        games = games.groupby('gameID', sort=False)
        slice_ratings = {}
        slice_entries = {}
        game = []
        for game_id, game in games:
            game = game.loc[:, ['gameID', 'userID', 'points_share', 'processTime']]
            self.glicko_calculator.process_glicko_game(game, slice_ratings, slice_entries)

        num_players_in_variant = len(game)
        if len(game) == 0:
            raise Exception("No games found for slice")
        ratings = pandas.DataFrame.from_dict(slice_ratings, orient='index',
                                             columns=['rating', 'confidence', 'volatility',
                                                      'rating_diffs', 'scores'])
        old_ratings_len = len(ratings)
        ratings = ratings.join(self.user_data, how='inner')
        assert old_ratings_len == len(ratings)
        if filter_inactive:
            time_limit = time.time() - (86400 * activity_filter)
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

        rating_med = ratings['rating'].median()
        confidence_med = ratings['confidence'].median()
        gc = self.glicko_calculator
        ratings['expected_score_vs_median'] = gc.expected_score_vs_player(ratings,
                                                                          rating_med,
                                                                          confidence_med)

        ratings_mean = ratings['rating'].mean()
        confidence_mean = ratings['confidence'].mean()
        ratings['expected_score_vs_mean'] = gc.expected_score_vs_player(ratings,
                                                                        ratings_mean,
                                                                        confidence_mean)

        ratings['expected_score_vs_new'] = gc.expected_score_vs_player(ratings,
                                                                       gc.init_rating,
                                                                       gc.init_conf)

        best_rating = ratings.iloc[0]['rating']
        best_confidence = ratings.iloc[0]['confidence']
        ratings['expected_score_vs_best'] = gc.expected_score_vs_player(ratings, best_rating,
                                                                        best_confidence)

        # ((1 / N) * (7-N)) / 6 = x, solve for N
        ratings['expected_draw_size_vs_mean'] = num_players_in_variant / (
                (ratings['expected_score_vs_mean'] * (num_players_in_variant - 1)) + 1)
        ratings['expected_score_in_mean_game'] = 1 / ratings['expected_draw_size_vs_mean']

        ratings = ratings[
            ['username', 'num_games', 'rating_lowerbound', 'rating', 'confidence',
             'expected_score_vs_median',
             'expected_score_vs_mean', 'expected_score_vs_best', 'expected_score_vs_new',
             'expected_score_in_mean_game', 'rating_diffs', 'rank_lb', 'rank_pure']]
        ratings.columns = ['name', 'num_games', 'rating_lb', 'rating', 'RD', 'vs_med',
                           'vs_mean', 'vs_best', 'vs_new', 'ppg_vs_mean', 'avg_rating_diff',
                           'rank_lb',
                           'rank_pure']

        ratings.set_index('name', drop=True, inplace=True)

        game_history = dict()
        for userID, games in slice_entries.items():
            user_name = self.user_data.ix[userID]['username']
            for game in games:
                game_id = game[0]
                end_rating = game[1]
                end_rd = game[2]
                timestamp = int(game[3])
                point_share = game[4]

                game_history[(user_name, timestamp, game_id)] = (
                    end_rating, end_rd, point_share)

        gc = self.glicko_calculator
        game_history = pandas.DataFrame.from_dict(game_history, orient='index')
        game_history.columns = ['rating', 'RD', 'point_share']
        game_history.index = pandas.MultiIndex.from_tuples(game_history.index.values)
        game_history['rating_diff'] = game_history['rating'].groupby(level=0).diff()

        game_history['rating_diff'] = game_history.apply(
            lambda row: row['rating'] - gc.init_rating if numpy.isnan(row['rating_diff']) else
            row['rating_diff'],
            axis=1)

        game_history['RD_diff'] = game_history['RD'].groupby(level=0).diff()
        game_history['RD_diff'] = game_history.apply(
            lambda row: row['RD'] - gc.init_conf if numpy.isnan(row['RD_diff']) else
            row['RD_diff'],
            axis=1)

        game_history.index.names = ['name', 'timestamp', 'gameid']

        return ratings, game_history
