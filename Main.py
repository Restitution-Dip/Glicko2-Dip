import pandas
from tqdm import tqdm
from datetime import datetime
from DataLoader import load_gr_data, get_history_and_user_data
import Config
import time
from CategoryResults import CategoryResults

# TODO: order by timestamp
chunks = []
tqdm.pandas()

pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import Panel, Tabs, TextInput, Button
from bokeh.io import show


def get_history_slice(history_data, variant_id: int = None, press_type: str = None,
                      phase_minutes_min: int = None, phase_minutes_max: int = None,
                      pot_type: str = None):
    # phaseMinutesMin and phaseMinutesMax are inclusive
    data = history_data
    if variant_id is not None:
        data = data[data['variantID'] == variant_id]

    if press_type is not None:
        data = data[data['pressType'] == press_type]

    if phase_minutes_min is not None:
        data = data[data['phaseMinutes'] >= phase_minutes_min]

    if phase_minutes_max is not None:
        data = data[data['phaseMinutes'] <= phase_minutes_max]

    if pot_type is not None:
        data = data[data['potType'] == pot_type]

    return data

main_start = time.time()
game_data, etag_pickle_folder = load_gr_data()
Config.ETAG_PICKLE_FOLDER = etag_pickle_folder
history_data, user_data = get_history_and_user_data(game_data, etag_pickle_folder)
most_recent_timestamp = history_data['processTime'].max()
most_recent_time = datetime.fromtimestamp(most_recent_timestamp)
print('Most recent game was on {}'.format(most_recent_time))
now = time.time()
now = datetime.fromtimestamp(now)
print('This script is being run on {}'.format(now))
print('It has been {} since the data source was updated.'.format(now - most_recent_time))

chaos = get_history_slice(history_data, variant_id=17)
classic = get_history_slice(history_data, variant_id=1)

# Classic FP
classic_fullpress = get_history_slice(classic,
                                      press_type='Regular')
classic_fullpress_live = get_history_slice(classic_fullpress,
                                           phase_minutes_max=60)
classic_fullpress_nl = get_history_slice(classic_fullpress,
                                         phase_minutes_min=61)

classic_fullpress_sos_nl = get_history_slice(classic_fullpress_nl,
                                             pot_type='Sum-of-squares'
                                             )

classic_fullpress_dss_nl = get_history_slice(classic_fullpress_nl,
                                             pot_type='Winner-takes-all')



classic_gb = get_history_slice(classic,
                               press_type='NoPress')

classic_gb_nl = get_history_slice(classic_gb,
                                  phase_minutes_min=61
)

classic_gb_sos_nl = get_history_slice(classic_gb_nl,
                                      pot_type='Sum-of-squares'
)

classic_gb_dss_nl = get_history_slice(classic_gb_nl,
                                      pot_type='Winner-takes-all'
)

classic_gb_live = get_history_slice(classic_gb,
                                  phase_minutes_max=60)



# Variants
chaos_results = CategoryResults('All Chaos',
                                chaos,
                                user_data,
                                placement_games=0
                                )

variant_results = [chaos_results]


c_fp_results = CategoryResults("All Classic Full Press",
                               classic_fullpress,
                               user_data
                               )
c_fp_nl_results = CategoryResults('Classic Full Press (Non-Live)',
                                  classic_fullpress_nl,
                                  user_data)
c_fp_sos_nl_results = CategoryResults('Classic Full Press (SoS) (Non-Live)',
                                      classic_fullpress_sos_nl,
                                      user_data)

c_fp_dss_nl_results = CategoryResults('Classic Full Press (DSS) (Non-Live)',
                                      classic_fullpress_dss_nl,
                                      user_data)

c_fp_live_results = CategoryResults('Classic Full Press (Live)',
                                    classic_fullpress_live,
                                    user_data)

classic_fp_results = [c_fp_results, c_fp_nl_results, c_fp_sos_nl_results, c_fp_dss_nl_results, c_fp_live_results]

c_gb_results = CategoryResults('All Classic Gunboat',
                               classic_gb,
                               user_data)
c_gb_sos_nl_results = CategoryResults('Classic Gunboat (SoS) (Non-Live)',
                                      classic_gb_sos_nl,
                                      user_data)
c_gb_dss_nl_results = CategoryResults('Classic Gunboat (DSS) (Non-Live)',
                                      classic_gb_dss_nl,
                                      user_data)
c_gb_live_results = CategoryResults('Classic Gunboat (Live)',
                                    classic_gb_live,
                                    user_data)

classic_gb_results = [c_gb_results, c_gb_sos_nl_results, c_gb_dss_nl_results, c_gb_live_results]


all_results = variant_results + classic_fp_results + classic_gb_results

# for result in all_results:
#     result.tabulate()
#     result.show_history()

def get_history_tabs(username):
    tabs = []

    classic_fp_tabs = []
    classic_gb_tabs = []
    variant_tabs = []
    for result in classic_fp_results:
        plot = result.get_history_plot(username)
        tab = Panel(child=plot, title=plot.title.text)
        classic_fp_tabs.append(tab)

    for result in classic_gb_results:
        plot = result.get_history_plot(username)
        tab = Panel(child=plot, title=plot.title.text)
        classic_gb_tabs.append(tab)

    for result in variant_results:
        plot = result.get_history_plot(username)
        tab = Panel(child=plot, title=plot.title.text)
        variant_tabs.append(tab)

    classic_fp_tabs = Panel(child=Tabs(tabs=classic_fp_tabs), title="Classic FP")
    classic_gb_tabs = Panel(child=Tabs(tabs=classic_gb_tabs), title="Classic GB")
    variant_tabs = Panel(child=Tabs(tabs=variant_tabs), title="Variants")
    main_tabs = Tabs(tabs=[classic_fp_tabs, classic_gb_tabs, variant_tabs], name="Player History")
    return main_tabs

def get_leaderboard_tabs():

    classic_fp_tabs = []
    classic_gb_tabs = []
    variant_tabs = []
    for result in classic_fp_results:
        plot = result.get_leaderboard()
        tab = Panel(child=plot, title=result.name)
        classic_fp_tabs.append(tab)

    for result in classic_gb_results:
        plot = result.get_leaderboard()
        tab = Panel(child=plot, title=result.name)
        classic_gb_tabs.append(tab)

    for result in variant_results:
        plot = result.get_leaderboard()
        tab = Panel(child=plot, title=result.name)
        variant_tabs.append(tab)

    classic_fp_tabs = Panel(child=Tabs(tabs=classic_fp_tabs), title="Classic FP")
    classic_gb_tabs = Panel(child=Tabs(tabs=classic_gb_tabs), title="Classic GB")
    variant_tabs = Panel(child=Tabs(tabs=variant_tabs), title="Variants")
    main_tabs = Tabs(tabs=[classic_fp_tabs, classic_gb_tabs, variant_tabs], name="Leaderboards")
    return main_tabs

leaderboard_tabs = get_leaderboard_tabs()

search_input = TextInput(title="Player to search:")


def handle_search():
    blah = curdoc().get_model_by_name("Player History")
    if blah is not None:
        layout.children.remove(blah)
    layout.children.append(get_history_tabs(search_input.value_input))
    return True


search_button = Button(label="Search")
search_button.on_click(handle_search)

layout = column(leaderboard_tabs, search_input, search_button)
# main_tabs = get_history_tabs(best)
curdoc().add_root(layout)

show(leaderboard_tabs)
# classic_fullpress_nl_ratings['best_rank'] = classic_fullpress_nl_ratings['GR_rank'].combine(
#     classic_fullpress_nl_ratings['rank_lb'], min, 0)
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings[
#     classic_fullpress_nl_ratings['best_rank'] <= 30]
# classic_fullpress_nl_ratings['abs_rank_diff'] = classic_fullpress_nl_ratings['rank_diff'].abs()
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['abs_rank_diff'],
#                                                                         ascending=False)
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.head(100)
# # classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['rank_diff'], ascending=False)
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings.sort_values(by=['rank_lb'],
#                                                                         ascending=True)
# classic_fullpress_nl_ratings = classic_fullpress_nl_ratings[
#     ['rank_diff', 'num_games', 'avg_rating_diff', 'rank_lb', 'GR_rank']]
# tabulate_ratings("Classic FP (Non-Live) Big Upsets",
#                  (classic_fullpress_nl_ratings, classic_fullpress_nl_entries), head=1000)

# for name in classic_fullpress_nl_ratings.index.values:
#     userID = user_data[user_data['username'] == name].index.values[0]
#     list_of_games = classic_fullpress_nl_entries[userID]
#     list_of_games = [(x[0], x[1], x[2], x[3], x[4]) for x in list_of_games]
#     list_of_diffs = []
#     prev_rat = 1500
#     for game in list_of_games:
#         rat = game[1]
#         p_share = game[4]
#         RD = game[2]
#         list_of_diffs.append((rat - prev_rat, p_share, RD))
#         prev_rat = rat
#
#     print(name, list_of_diffs)
