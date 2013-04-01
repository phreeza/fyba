from pymc import Exponential, deterministic, Poisson, Normal, Deterministic, Uniform
import numpy as np


class LeagueModel(object):
    """MCMC model of a football league."""
    
    #TODO: Kelly Bettor
    #TODO: refine model
    #TODO: identify columns
    
    def __init__(self, fname, playedto=None):
        super(LeagueModel, self).__init__()
        league = League(fname,playedto)

        N = len(league.teams)
        def outcome_eval(home=None,away=None):
            if home > away:
                return 1
            if home < away:
                return -1
            if home == away:
                return 0
            
        def clip_rate(val):
            if val>0.2:return val
            else: return 0.2
        
        
        self.goal_rate = np.empty(N,dtype=object)
        self.def_rate = np.empty(N,dtype=object)
        self.match_rate = np.empty(len(league.games)*2,dtype=object)
        self.outcome_future = np.empty(len(league.games),dtype=object)
        self.match_goals_future = np.empty(len(league.future_games)*2,dtype=object)
        self.home_adv = Uniform(name = 'home_adv',lower=0.,upper=0.7)
        self.league = league

        for t in league.teams.values():
            self.goal_rate[t.team_id] = Exponential('goal_rate_%i'%t.team_id,beta=1)
            self.def_rate[t.team_id] = Exponential('def_rate_%i'%t.team_id,beta=1)

        for game in range(len(league.games)):
            self.match_rate[2*game] = Poisson('match_rate_%i'%(2*game),
                    mu=Deterministic(eval=clip_rate,
                                     parents={'val':
                                        self.goal_rate[league.games[game].hometeam.team_id] - 
                                        self.def_rate[league.games[game].awayteam.team_id] + self.home_adv},
                                     doc='clipped goal rate',name='clipped_h_%i'%game),
                    value=league.games[game].homescore, observed=True)
            self.match_rate[2*game+1] = Poisson('match_rate_%i'%(2*game+1),
                    mu=Deterministic(eval=clip_rate,
                                     parents={'val':
                                        self.goal_rate[league.games[game].awayteam.team_id] - 
                                        self.def_rate[league.games[game].hometeam.team_id]},
                                     doc='clipped goal rate',name='clipped_a_%i'%game),
                    value=league.games[game].awayscore, observed=True)


        for game in range(len(league.future_games)):
            self.match_goals_future[2*game] = Poisson('match_goals_future_%i_home'%game,
                    mu=Deterministic(eval=clip_rate,
                                     parents={'val':
                                        self.goal_rate[league.future_games[game][0].team_id] - 
                                        self.def_rate[league.future_games[game][1].team_id] + self.home_adv},
                                     doc='clipped goal rate',name='clipped_fut_h_%i'%game))

            self.match_goals_future[2*game+1] = Poisson('match_goals_future_%i_away'%game,
                    mu=Deterministic(eval=clip_rate,
                                     parents={'val':
                                        self.goal_rate[league.future_games[game][1].team_id] - 
                                        self.def_rate[league.future_games[game][0].team_id]},
                                     doc='clipped goal rate',name='clipped_fut_a_%i'%game))

            self.outcome_future[game] = Deterministic(eval=outcome_eval,parents={
                'home':self.match_goals_future[2*game],
                'away':self.match_goals_future[2*game+1]},name='match_outcome_future_%i'%game,
                dtype=int,doc='The outcome of the match'
                )
            
    def run_mc(self,nsample = 10000,interactive=False):
        """run the model using mcmc"""
        from pymc import MCMC
        self.M = MCMC(self)
        if interactive:
            self.M.isample(iter=nsample, burn=1000, thin=10)
        else:
            self.M.sample(iter=nsample, burn=1000, thin=10)
        #plot(self.M)

class Prediction(object):
    """A prediction of outcomes of a group of games"""
    def __init__(self, league, outcome_future):
        self.predictions = []
        for n,g in enumerate(league.future_games):
            g = list(g)
            g.append(float((outcome_future[n].trace()==1).sum())/len(outcome_future[n].trace()))
            g.append(float((outcome_future[n].trace()==0).sum())/len(outcome_future[n].trace()))
            g.append(float((outcome_future[n].trace()==-1).sum())/len(outcome_future[n].trace()))
            self.predictions.append(g)
    
        self.edges = []
        for q in self.predictions:
            if q[3]-q[6]<0:self.edges.append((q[6]-q[3],self.kellybet(q[3],q[6]),q[2]=='H',q[3]))
            if q[4]-q[7]<0:self.edges.append((q[7]-q[4],self.kellybet(q[4],q[7]),q[2]=='D',q[4]))
            if q[5]-q[8]<0:self.edges.append((q[8]-q[5],self.kellybet(q[5],q[8]),q[2]=='A',q[5]))
    
    def kellybet(self,odds,prob ):
        return (prob/odds-1)/(1./odds-1)
    
    def returns(self):
        ret = 0.
        for b in self.edges:
            if b[2]:
                ret += b[1]*(1./b[3]-1.)
            else:
                ret -= b[1]
                
        return ret
        
    
    
    
    

class Team(object):
    """Representation of a Team"""
    def __init__(self, name):
        self.name = name
        self.team_id = -1

    def __repr__(self):
        """represent this object with its name"""
        return "Team(\"%s\")" % self.name

    def __str__(self):
        "return team name"
        return self.name

class Game():
    """A game played between two teams"""
    def __init__(self, hometeam, awayteam, homescore, awayscore):
        (self.hometeam, self.awayteam, self.homescore, self.awayscore) = (hometeam,
                awayteam, homescore, awayscore)
    def __str__(self):
        return "Game %s - %s (%i:%i)" % (self.hometeam, self.awayteam,
                self.homescore, self.awayscore)
class League():
    """
    The league contains the teams that play in it and the games played.

    >>> league = League("csv/0001/D1.csv")
    >>> league.teams # doctest: +NORMALIZE_WHITESPACE
    {'Cottbus': Team("Cottbus"), 
            'Wolfsburg': Team("Wolfsburg"),
            'Leverkusen': Team("Leverkusen"),
            'Dortmund': Team("Dortmund"),
            'Hertha': Team("Hertha"),
            'Kaiserslautern': Team("Kaiserslautern"),
            'Schalke 04': Team("Schalke 04"),
            'Stuttgart': Team("Stuttgart"),
            'Bochum': Team("Bochum"),
            'Munich 1860': Team("Munich 1860"), 
            'Hamburg': Team("Hamburg"), 
            'Freiburg': Team("Freiburg"), 
            'Ein Frankfurt': Team("Ein Frankfurt"), 
            'Bayern Munich': Team("Bayern Munich"), 
            'Werder Bremen': Team("Werder Bremen"), 
            'FC Koln': Team("FC Koln"), 
            'Hansa Rostock': Team("Hansa Rostock"), 
            'Unterhaching': Team("Unterhaching")}
    """
    def __init__(self, fname, playedto=None):
        csv_file = file(fname)
        data = []
        for line in csv_file.readlines():
            data.append(line.split(','))
        teamnames = set(t[3] for t in data[1:])
        self.teams = dict((t,Team(t)) for t in teamnames)
        self.n_teams = len(self.teams)
        self.n_days = 2*(len(data)-1)/self.n_teams
        if playedto is None:
            playedto = len(data)-1
        else:
            playedto *= len(teamnames)/2
        index = 0
        for i in self.teams.values():
            i.team_id = index
            index += 1
        self.games = []
        for gameline in data[1:playedto+1]:
            self.games.append(Game(
                self.teams[gameline[2]],self.teams[gameline[3]],
                int(gameline[4]), int(gameline[5])
                ))

        self.future_games = []
        if playedto < len(data)-1-len(self.teams)/2:
            for gameline in data[playedto+1:playedto+1+len(self.teams)/2]:
                self.future_games.append(
                    [self.teams[gameline[2]],self.teams[gameline[3]],gameline[6],
                     1./float(gameline[22]),1./float(gameline[23]),1./float(gameline[24])]
                    )
                
                
def evaluate(fname='csv/1011/D1.csv'):
    values = []
    for n in range(3,34):
        l = LeagueModel(fname,n)
        l.run_mc()
        p = Prediction(l.league,l.outcome_future)
        values.append(p.returns())
        print values
    return values
