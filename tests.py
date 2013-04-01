import fyba

def simple_test():
    l = fyba.LeagueModel('csv/1011/D1.csv')
    l.run_mc(nsample=1000)
    
def playedto_test():
    l = fyba.LeagueModel('csv/1011/D1.csv',playedto = 30)
    l.run_mc(nsample=1000)
