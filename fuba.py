'''
File: fuba.py
Author: Thomas McColgan
Description: Class structure for my football simulation
'''

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
        if playedto is None:
            playedto = len(data)-1
        teamnames = set(t[3] for t in data[1:])
        self.teams = dict((t,Team(t)) for t in teamnames)
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
        if playedto < len(data)-1:
            for gameline in data[playedto+1:]:
                self.future_games.append(
                    [self.teams[gameline[2]],self.teams[gameline[3]]]
                    )