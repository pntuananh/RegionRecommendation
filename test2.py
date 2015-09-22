dataset = 'checkin2011'
inputfile = 'FoursquareCheckins20110101-20111231.csv'

#dataset = 'SHTiesData'
#inputfile = 'FoursquareCheckins.csv'

from header import *
from collections import defaultdict, Counter
import random, pdb, math

from bisect import bisect_right, bisect_left
import numpy as np
import networkx as nx

import warnings
warnings.filterwarnings('error')

R = 5 
delta_lon = 0.012 * R
delta_lat = 0.009 * R

checkins = defaultdict(set)
poi_info = {}
poi_users = defaultdict(set)

f = open(dataset + '\\' + inputfile)
f.readline()

c = 0
for line in f:
    if not line: continue
    userid, lat, lon, time, poi = line.strip('\n').split(',')
    lat = float(lat)
    lon = float(lon)

    checkins[userid].add(poi)

    if poi not in poi_info:
        poi_info[poi] = (lon,lat)

        #lat_poi += [(lat,poi)]
        #lon_poi += [(lon,poi)]

    poi_users[poi].add(userid)

    c += 1
    if c%100000 == 0:
        print '\r%d' % c,

print ''
print c
print len(checkins)
print len(poi_info)

#lines = ['%s %f %f' % (poi,lat,lon) for poi,(lat,lon) in poi_info.iteritems()] 
#f = open('locations.txt', 'w')
#f.write('\n'.join(lines))
#f.close()
#pdb.set_trace()


cont = 1
while cont:
    cont = 0
    for user in checkins.keys():
        if len(checkins[user]) < 5:
            for poi in checkins[user]:
                poi_users[poi].remove(user)

            del checkins[user]
            cont = 1

    for poi in poi_users.keys():
        if len(poi_users[poi]) < 5:
            for user in poi_users[poi]:
                checkins[user].remove(poi)

            del poi_users[poi]
            cont = 1

print len(checkins)
print len(poi_users)

def get_hometown():
    user_hometown = {}
    hometown_user = defaultdict(list)
    f = open(dataset + '\\hometown.csv' )
    f.readline()

    for line in f:
        user, hometown = line.strip('\n').split(',')

        user_hometown[user] = hometown
        hometown_user[hometown].append(user)

    poi_hometown = {}
    hometown_poi = {}

    f = open('hometown_poi.txt')
    for line in f:
        parts = line.strip('\n').split()
        hometown = parts[0]
        pois = parts[1:]
        hometown_poi[hometown] = set(pois)
        for p in pois:
            poi_hometown[p] = hometown
        
    return user_hometown, hometown_user, poi_hometown, hometown_poi 

user_hometown, hometown_user, poi_hometown, hometown_poi = get_hometown()

coordinates = {}
print 'reading locations...'
coordinates = {}
for parts in my_open('locations.txt'):
    poi, lon, lat = parts
    coordinates[poi] = (float(lon),float(lat))

poi_users = defaultdict(set)
training = {}
testing = {}

print 'creating training and testing sets...'
all_users = checkins.keys()
random.shuffle(all_users)

for user in all_users:
    poi_set = checkins[user]

    to_training = 1
    if len(testing) < 2000:
        outtown_counter = Counter([poi_hometown[p] for p in poi_set if user_hometown[user] != poi_hometown[p]])
        outtowns = [t for t,c in outtown_counter.iteritems() if c >= 3]
        if len(outtowns) > 1:
            for ot in outtowns:
                outside_pois = set([p for p in poi_set if poi_hometown[p] != ot])
                if len(outside_pois) < 5:
                    continue

                inside_pois = [p for p in poi_set if poi_hometown[p] == ot]
                
                edges = []
                for i1 in range(len(inside_pois)-1):
                    p1 = inside_pois[i1]
                    lon1, lat1 = coordinates[p1]
                    for i2 in range(i1+1, len(inside_pois)):
                        p2 = inside_pois[i2]
                        lon2, lat2 = coordinates[p2]

                        if abs(lon1-lon2) <= delta_lon and abs(lat1-lat2) <= delta_lat:
                            edges += [(p1,p2)]


                if not edges:
                    continue
                else:
                    training[user] = outside_pois
                    testing[user] = (ot, inside_pois)

                    to_training = 0
                    break

    if to_training:
        training[user] = poi_set
        
for user, poi_set in training.iteritems():
    for poi in poi_set:
        poi_users[poi].add(user)

print len(training), len(testing)

f = open('training.txt', 'w')
lines = ['%s %s' % (u, ' '.join(training[u])) for u in training]
f.write('\n'.join(lines))
f.close()

f = open('testing.txt', 'w')
lines = []
for u in testing:
    ot, pois = testing[u]
    line = [u, ot, ' '.join(pois)]
    lines.append(' '.join(line))

f.write('\n'.join(lines))
f.close()
