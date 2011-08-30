'''
File: fuba.py
Author: Thomas McColgan
Description: Class structure for my football simulation
'''

class Team(object):
    """Representation of a Team"""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        """represent this object with its name"""
        return self.name

class Game():
    """A game played between two teams"""
    def __init__(self, hometeam, awayteam, homescore, awayscore):
        (self.hometeam, awayteam, homescore, awayscore) = (hometeam, 
                awayteam, homescore, awayscore)
class League():
    """
    The league contains the teams that play in it and the games played.

    >>> league = League("csv/0001/D1.csv")
    >>> league.teams
    {'Cottbus': Cottbus, 'Wolfsburg': Wolfsburg, 'Leverkusen': Leverkusen, 'Dortmund': Dortmund, 'Hertha': Hertha, 'Kaiserslautern': Kaiserslautern, 'Schalke 04': Schalke 04, 'Stuttgart': Stuttgart, 'Bochum': Bochum, 'Munich 1860': Munich 1860, 'Hamburg': Hamburg, 'Freiburg': Freiburg, 'Ein Frankfurt': Ein Frankfurt, 'Bayern Munich': Bayern Munich, 'Werder Bremen': Werder Bremen, 'FC Koln': FC Koln, 'Hansa Rostock': Hansa Rostock, 'Unterhaching': Unterhaching}
    """
    def __init__(self, fname):
        csv_file = file(fname)
        data = []
        for line in csv_file.readlines():
            data.append(line.split(','))
        teamnames = set(t[3] for t in data[1:])
        self.teams = dict((t,Team(t)) for t in teamnames)
