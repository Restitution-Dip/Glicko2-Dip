from Config import *

_MAXSIZE = 186
_MAXMULTI = .272
_MULTISLOPE = .00391
_WIN = 1.0
_LOSS = 0
_CATCH = .5
_INITRAT = 1500
_INITCONF = 100.0
_TYPICAL_CONF = 15
_CONF_INC_PER_MONTH = math.sqrt((_INITCONF ** 2 - _TYPICAL_CONF ** 2) / 100)
_INITVOL = .005
_VOL = .1
_CONV = 173.7178
_EPS = 0.001


class GlickoCalculator:
    def __init__(self,
                 max_size=_MAXSIZE,
                 max_multi=_MAXMULTI,
                 multi_slope=_MULTISLOPE,
                 win=_WIN,
                 loss=_LOSS,
                 catch=_CATCH,
                 init_rating=_INITRAT,
                 init_conf=_INITCONF,
                 typical_conf=_TYPICAL_CONF,
                 conf_inc_per_month=_CONF_INC_PER_MONTH,
                 init_vol=_INITVOL,
                 vol=_VOL,
                 conv=_CONV,
                 eps=_EPS
                 ):
        self.max_size = max_size
        self.max_multi = max_multi
        self.multi_slope = multi_slope
        self.win = win
        self.loss = loss
        self.catch = catch
        self.init_rating = init_rating
        self.init_conf = init_conf
        self.typical_conf = typical_conf
        self.conf_inc_per_month = conf_inc_per_month
        self.init_vol = init_vol
        self.vol = vol
        self.conv = conv
        self.eps = eps

    def expected_score_vs_player(self, ratings, opp_rating, opp_confidence):
        opp_mu = (opp_rating - self.init_rating) / _CONV
        opp_phi = opp_confidence / _CONV
        opp_weight = 1 / math.sqrt(1 + 3 * (opp_phi ** 2) / (math.pi ** 2))
        all_mu_diff = -opp_weight * (((ratings['rating'] - _INITRAT) / _CONV) - opp_mu)
        all_mu_diff = all_mu_diff.apply(math.exp)
        expected_score_vs_opp = 1 / (1 + all_mu_diff)
        expected_score_vs_opp = (expected_score_vs_opp - 0.5) * 2
        return expected_score_vs_opp

    def calculate_change_over_time(self, players, entries, process_time):
        for player in players:
            if player.userID in entries:
                time_since_last_game = process_time - entries[player.userID][-1][3]
                months_since_last_game = time_since_last_game / 2.592e+6
                if months_since_last_game > 1:
                    new_confidence = min(self.init_conf,
                                         math.sqrt(
                                             player.confidence ** 2 +
                                             self.conf_inc_per_month ** 2 * months_since_last_game))
                    player.confidence = new_confidence

    def calculate_glicko(self, players):
        INIT_RATING = self.init_rating
        CONV = self.conv

        N = len(players)
        if N > self.max_size:
            multi = self.max_multi
        else:
            multi = self.win - self.multi_slope * N
        # compare every head to head matchup in a given compeition
        for i in players:
            mu = (i.rating - INIT_RATING) / CONV
            phi = i.confidence / CONV
            sigma = i.volatility
            v_inv = 0
            delta = 0
            for j in players:
                if i is not j:
                    oppMu = (j.rating - INIT_RATING) / CONV
                    i.rating_diffs.append(i.rating - j.rating)
                    oppPhi = j.confidence / CONV

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
                newSigma = self.findSigma(mu, phi, sigma, change, v)
                phiAst = math.sqrt(phi ** 2 + newSigma ** 2)
                # New confidence based on competitors volatility and v
                newPhi = 1 / math.sqrt(1 / phiAst ** 2 + 1 / v)

                newMu = mu + newPhi ** 2 * delta
                i.rating = newMu * CONV + INIT_RATING
                i.confidence = newPhi * CONV
                i.volatility = newSigma

    def process_glicko_game(self, game, ratings, entries):
        INIT_RATING = self.init_rating
        INIT_CONF = self.init_conf
        INIT_VOL = self.init_vol
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
                meet.addCompetitor(userID, points_share, INIT_RATING, INIT_CONF, INIT_VOL, [])
        if len(meet.competitors) > 1:
            self.calculate_change_over_time(meet.competitors, entries, process_time)
            self.calculate_glicko(meet.competitors)

            # Take results of competition and append data

            for player in meet.competitors:
                ratings[player.userID] = [player.rating, player.confidence, player.volatility,
                                          player.rating_diffs, player.scores]
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

    def get_expected_score(self, p1_rating, p2_rating, p2_confidence):
        INIT_RATING = self.init_rating
        CONV = self.conv

        mu = (p1_rating - INIT_RATING) / CONV
        oppMu = (p2_rating - INIT_RATING) / CONV
        oppPhi = p2_confidence / CONV

        weighted = 1 / math.sqrt(1 + 3 * (oppPhi ** 2) / (math.pi ** 2))
        # weighted = 1
        expected_score = 1 / (1 + math.exp(-weighted * (mu - oppMu)))
        if USE_POINT_DIFFERENCE:
            expected_score = expected_score - 0.5
            expected_score = expected_score * 2
        return expected_score


    def get_expected_score_by_username(self, ratings, userid_1, userid_2):
        INIT_RATING = self.init_rating
        CONV = self.conv

        ratings_1 = ratings.loc[ratings['username'] == userid_1]
        ratings_2 = ratings.loc[ratings['username'] == userid_2]
        mu = (ratings_1['rating'].iloc[0] - INIT_RATING) / CONV
        oppMu = (ratings_2['rating'].iloc[0] - INIT_RATING) / CONV

        return self.get_expected_score(ratings_1['rating'].iloc[0],
                                  ratings_2['rating'].iloc[0],
                                  ratings_2['confidence'].iloc[0])

    def findSigma(self, mu, phi, sigma, change, v):
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


