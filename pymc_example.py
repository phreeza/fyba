from pymc import Exponential, deterministic, Poisson, Normal
import numpy as np
import fuba


class LeagueModel(object):
    """MCMC model of a football league."""
    def __init__(self, fname):
        super(LeagueModel, self).__init__()
        league = fuba.League(fname)

        goals_array = np.array([ 0, 5, 4, 0, 1, 4, 3, 4,
            0, 6, 3, 3, 4, 0, 2, 6,3 , 3, 5, 4, 5, 3, 1, 4,
            4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
            2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1,
            0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
            1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1,
            1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0, 1])

        N = 18
        self.goal_rate = np.empty(N,dtype=object)
        self.match_rate = np.empty(len(goals_array),dtype=object)
        self.home_adv = Normal(name = 'home_adv',mu=0,tau=10.)


        for team in range(N):
            self.goal_rate[team] = Exponential('goal_rate_%i'%team,beta=1)

        for game in range(len(goals_array)):
            if game%N == 0:
                goals_array[game] = 0
            if (game/N)%2 == 0:
                self.match_rate[game] = Poisson('match_rate_%i'%game,
                        mu=self.goal_rate[game%N],
                        value=goals_array[game], observed=True)
            else:
                self.match_rate[game] = Poisson('match_rate_%i'%game,
                        mu=self.goal_rate[game%N] + self.home_adv,
                        value=goals_array[game], observed=True)


    def run_mc(self,nsample = 10000):
        """run the model using mcmc"""
        from pymc.Matplot import plot
        from pymc import MCMC
        M = MCMC(self)
        M.sample(iter=nsample, burn=1000, thin=10)
        plot(M)
