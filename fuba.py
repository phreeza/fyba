'''
File: fuba.py
Author: Thomas McColgan
Description: Class structure for my football simulation
'''

class Team(object):
    """Representation of a Team"""
    def __init__(self, name):
        self.name = name

class Game():
    """A game played between two teams"""
    def __init__(self, hometeam, awayteam, homescore, awayscore):
        (self.hometeam, awayteam, homescore, awayscore) = (hometeam, 
                awayteam, homescore, awayscore)
class League():
    """The league contains the teams that play in it and the games pleyed."""
    def __init__(self, fname):
        csv_file = file(fname)
        data = []
        for line in csv_file.readlines():
            data.append(line.split(','))
        teamnames = set(t[3] for t in data[1:])
        self.teams = dict((t,team(t)) for t in teamnames)
