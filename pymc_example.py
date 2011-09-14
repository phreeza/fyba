from pymc import Exponential, deterministic, Poisson, Normal
import numpy as np
import fuba


class LeagueModel(object):
    """MCMC model of a football league."""
    def __init__(self, fname):
        super(LeagueModel, self).__init__()
        league = fuba.League(fname)

        N = len(league.teams)
        #dummy future games
        future_games = [[league.teams["Werder Bremen"],league.teams["Hamburg"]]]

        self.goal_rate = np.empty(N,dtype=object)
        self.match_rate = np.empty(len(league.games)*2,dtype=object)
        self.match_goals_future = np.empty(len(future_games)*2,dtype=object)
        self.home_adv = Normal(name = 'home_adv',mu=0,tau=10.)

        for t in league.teams.values():
            print t.name,t.team_id
            self.goal_rate[t.team_id] = Exponential('goal_rate_%i'%t.team_id,beta=1)

        for game in range(len(league.games)):
            self.match_rate[2*game] = Poisson('match_rate_%i'%(2*game),
                    mu=self.goal_rate[league.games[game].hometeam.team_id] + self.home_adv,
                    value=league.games[game].homescore, observed=True)
            self.match_rate[2*game+1] = Poisson('match_rate_%i'%(2*game+1),
                    mu=self.goal_rate[league.games[game].hometeam.team_id],
                    value=league.games[game].homescore, observed=True)

        for game in range(len(future_games)):
            self.match_goals_future[2*game] = Poisson('match_goals_future_%i'%(2*game),
                    mu=self.goal_rate[future_games[game][0].team_id] + self.home_adv)
            self.match_goals_future[2*game+1] = Poisson('match_goals_future_%i'%(2*game+1),
                    mu=self.goal_rate[future_games[game][1].team_id])

    def run_mc(self,nsample = 10000):
        """run the model using mcmc"""
        from pymc.Matplot import plot
        from pymc import MCMC
        M = MCMC(self)
        M.sample(iter=nsample, burn=1000, thin=10)
        plot(M)
