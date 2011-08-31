from pymc import DiscreteUniform, Exponential, deterministic, Poisson
import numpy as np

goals_array = np.array([ 4, 5, 4, 0, 1, 4, 3, 4,
    0, 6, 3, 3, 4, 0, 2, 6,3 , 3, 5, 4, 5, 3, 1, 4,
    4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2,
    2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1,
    0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
    1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1,
    1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 0, 1])

N = 18
goal_rate = np.empty(N,dtype=object)
match_rate = np.empty(len(goals_array),dtype=object)

for team in range(N):
    goal_rate[team] = Exponential('goal_rate_%i'%team,beta=1)

for game in range(len(goals_array)):
    match_rate[game] = Poisson('match_rate_%i'%game, mu=goal_rate[game%N],
            value=goals_array[game], observed=True)
