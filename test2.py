dataset = 'checkin2011'
inputfile = 'FoursquareCheckins20110101-20111231.csv'

from collections import defaultdict, Counter
import itertools
import numpy as np

from header import *
import pdb

import networkx as nx

regular = 0.03

R = 5
delta_lat = 0.009 
delta_lon = 0.012

coordinates = {}
print 'reading locations...'
coordinates = {}
for parts in my_open('locations.txt'):
    poi, lon, lat = parts
    coordinates[poi] = (float(lon),float(lat))


def find_near_pois(center, r=R):
    x, y = center
    #s = set()
    R_lon = delta_lon*r/2
    R_lat = delta_lat*r/2
    points_in_region = set([item.object for item in r_tree.intersection((x-R_lon, y-R_lat, x+R_lon, y+R_lat), objects=True)])

    return points_in_region
    #return set(p for p in s1&s2 if haversine(center,coordinates[p]) <= r)


def sample_cover_region(pois, r=R):
    lons = [coordinates[p][0] for p in pois]
    lats = [coordinates[p][1] for p in pois]

    R_lon = delta_lon*r/2
    R_lat = delta_lat*r/2

    min_lon = max(lons) - R_lon
    max_lon = min(lons) + R_lon
    min_lat = max(lats) - R_lat
    max_lat = min(lats) + R_lat

    x = min_lon + (max_lon-min_lon)*np.random.random() 
    y = min_lat + (max_lat-min_lat)*np.random.random() 

    points_in_region = set([item.object for item in r_tree.intersection((x-R_lon, y-R_lat, x+R_lon, y+R_lat), objects=True)])

    assert all(p in points_in_region for p in pois)

    return points_in_region


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

poi_users = defaultdict(set)
training = {}
testing = {}

f = open('training.txt')
for line in f:
    p = line.strip('\n').split()
    user = p[0]
    training[user] = set(p[1:])
f.close()

f = open('testing.txt')
for line in f:
    p = line.strip('\n').split()
    user = p[0]
    ot = p[1]
    pois = set(p[2:])

    testing[user] = (ot,pois)

for user, poi_set in training.iteritems():
    for poi in poi_set:
        poi_users[poi].add(user)

print 'reading preferences...'
user_pref = {}
user_pref_d = {}
poi_pref = {}

for p in my_open('model/user_factors_%f_1.txt' % regular):
    user = p[0]
    user_pref[user] = np.array(map(float,p[1:]))

for p in my_open('model/user_factors_d_%f_1.txt' % regular):
    user = p[0]
    user_pref_d[user] = np.array(map(float,p[1:]))

for p in my_open('model/poi_factors_%f_1.txt' % regular):
    poi = p[0]
    poi_pref[poi] = np.array(map(float,p[1:]))

global_user_sim = {}
f = open('user_sim.txt')
for line in f:
    p = line.strip('\n').split()
    test_user = p[0]
    global_user_sim[test_user] = {}
    for u_s in p[1:]:
        u,s = u_s.split('|')
        global_user_sim[test_user][u] = float(s)
f.close()

import rtree
print 'inserting to Rtree...'
r_tree = rtree.index.Index()
for p in poi_users:
    x,y = coordinates[p]
    r_tree.insert(0, (x,y,x,y), obj=p)


T = [1,3,5,10,20,50,-1,-3,-5,-20,-50]
pre = {}
pre_Q = {}
pre_Q_p = {}
pre_pop = {}
pre_CF = {}

for t in T:
    pre[t]     = []
    pre_Q[t]   = []
    pre_Q_p[t] = []
    pre_pop[t] = []
    pre_CF[t]  = []


score_parts = [0,0]
avg_prob = (0,0)

use_Q = 1
if use_Q:
    print 'computing Q...'
    lines = []
    c = 0
    Q = defaultdict(dict)

    f = open('q_2.txt')
    for line in f:
        p,v,w = line.split()
        # p --> v
        Q[p][v] = float(w)

        c += 1
        if c%100000 == 0:
            print '\r%d' % c,
    f.close()
    print ''

    def get_new_score(reg, score, top_pois=set()):
        global avg_prob
        lamb = 0.5
        new_score = {} #defaultdict(float)
        total_weights = defaultdict(list)

        
        for p in reg:
            if top_pois and p not in top_pois: continue

            sc = score.get(p,0.0)

            for sim_p in Q[p]:
                if sim_p not in reg or sim_p == p:
                    continue

                total_weights[sim_p] += [(Q[p][sim_p], sc)]
                avg_prob = (avg_prob[0] + Q[p][sim_p], avg_prob[1]+1)

        for p in reg:
            if p in total_weights:
                temp = sum(w*s for w,s in total_weights[p])
                temp /= sum(w for w,s in total_weights[p])

                part1, part2 = (1-lamb)*score.get(p,0.0) , lamb*temp
                #part1, part2 = score.get(p,0.0) , temp

                new_score[p] = part1 + part2

                score_parts[0] += part1
                score_parts[1] += part2

            else:
                new_score[p] = (1-lamb) * score.get(p,0.0) 

        return new_score

print ''

def predict(regions, t, true_regions):
    reg_score = {}
    for i, reg in enumerate(regions):
        if not reg: continue

        if t > 0:
            top_pois = [s for s,p in reg[:t]]
            reg_score[i] = sum(top_pois) / len(top_pois)
        elif t < 0:
            t = -t
            top_pois = [s for s,p in reg[:t]]
            #weights = [1.0/(j+1) for j in range(len(top_pois))]
            weights = [1.0/(j+1) for j in range(t)]
            reg_score[i] = sum(s*w for s,w in zip(top_pois, weights)) / sum(weights)

    ranked_reg = sorted([r for r in reg_score], key=lambda x : reg_score[x], reverse=True)
    correct = ranked_reg[0] in true_regions

    return correct


def most_pois(regions, true_regions):
    reg_score = {}
    for i, reg in enumerate(regions):
        reg_score[i] = reg

    ranked_reg = sorted([r for r in reg_score], key=lambda x : reg_score[x], reverse=True)
    correct = ranked_reg[0] in true_regions

    return correct


c = 0
print 'start...'
lines = []
debug = [0,0]
for target_user in testing:
    ot, true_pois = testing[target_user]

    score = defaultdict(float)
    for p in hometown_poi[ot]:
        score[p] = (user_pref[target_user] + user_pref_d[target_user]).dot(poi_pref[p])

        debug[score[p] < 0] += 1

    score_CF = defaultdict(float)
    for p in hometown_poi[ot]:
        s = 0
        for user in poi_users[p]:
            s += global_user_sim[target_user].get(user,0)

        if s > 0:
            score_CF[p] = s

    g_top_pois = set(sorted([p for p in hometown_poi[ot]], key=lambda x:len(poi_users[x]), reverse=True)[:5])
    #p_top_pois = set(sorted([p for p in hometown_poi[ot]], key=lambda x:len(poi_users[x]), reverse=True)[:5])
    p_top_pois = set(sorted([p for p in hometown_poi[ot]], key=lambda x:score.get(x,0), reverse=True)[:5])
    assert len(true_pois) >= 3

    #sample regions
    regions     = []
    regions_Q   = []
    regions_Q_p = []
    regions_N   = []
    regions_CF  = []

    # sample true regions
    max_n_pos = 0
    true_regions = set()

    edges = []
    for p1,p2 in itertools.combinations(true_pois,2):
        lon1,lat1 = coordinates[p1]
        lon2,lat2 = coordinates[p2]

        if abs(lon1-lon2) <= delta_lon*R and abs(lat1-lat2) <= delta_lat*R:
            edges += [(p1,p2)]

    G = nx.Graph()
    G.add_edges_from(edges)
    cliques = list(nx.find_cliques(G))
    assert len(cliques)

    cliques.sort(key=lambda x: len(x), reverse=True)
    seen = set()

    j = 0
    for cl in cliques:
        if any(p in seen for p in cl): continue

        reg = sample_cover_region(cl)

        sorted_r = sorted([(score.get(p,0),p) for p in reg], reverse=True)
        regions += [sorted_r]

        new_score = get_new_score(reg, score, g_top_pois)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        new_score = get_new_score(reg, score, p_top_pois)
        regions_Q_p += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        regions_N += [len(reg)]
        if (not true_regions) or (len(cl) == max_n_pos):
            true_regions.add(j)
            max_n_pos = len(cl)

        j += 1
        seen.update(cl)

        sorted_r_CF = sorted([(score_CF.get(p,0),p) for p in reg], reverse=True)
        regions_CF += [sorted_r_CF]


    for poi in true_pois - seen:
        reg = sample_cover_region([poi])

        sorted_r = sorted([(score.get(p,0),p) for p in reg], reverse=True)
        regions += [sorted_r]

        new_score = get_new_score(reg, score, g_top_pois)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        new_score = get_new_score(reg, score, p_top_pois)
        regions_Q_p += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        regions_N += [len(reg)]

        if (not true_regions):
            true_regions.add(j)
            max_n_pos = 1

        j += 1
        
        sorted_r_CF = sorted([(score_CF.get(p,0),p) for p in reg], reverse=True)
        regions_CF += [sorted_r_CF]


    #if not regions: continue

    neg_pois = [p for p in hometown_poi[ot] if p not in true_pois]

    np.random.shuffle(neg_pois)

    has_neg_region = 0
    for p in neg_pois:
        reg = find_near_pois(coordinates[p])

        if true_pois & reg:
            continue

        regions += [sorted([(score.get(p,0),p) for p in reg], reverse=True)]

        new_score = get_new_score(reg, score, g_top_pois)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        new_score = get_new_score(reg, score, p_top_pois)
        regions_Q_p += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        regions_N += [len(reg)]
        
        sorted_r_CF = sorted([(score_CF.get(p,0),p) for p in reg], reverse=True)
        regions_CF += [sorted_r_CF]

        has_neg_region += 1

        if has_neg_region == 20: 
            break

    #if len(regions) == 1:
    #    continue

    for t in T:
        if predict(regions, t, true_regions):
            pre[t] += [1.0]
        else:
            pre[t] += [0.0]

        if predict(regions_Q, t, true_regions):
            pre_Q[t] += [1.0]
        else:
            pre_Q[t] += [0.0]

        if predict(regions_Q_p, t, true_regions):
            pre_Q_p[t] += [1.0]
        else:
            pre_Q_p[t] += [0.0]

        if most_pois(regions_N, true_regions):
            pre_pop[t] += [1.0]
        else:
            pre_pop[t] += [0.0]

        if predict(regions_CF, t, true_regions):
            pre_CF[t] += [1.0]
        else:
            pre_CF[t] += [0.0]


    c += 1
    if c%10 == 0:
        print '\r%d' % c,
        #print '\r%d %f %f %f %f' % (c, avg_prob[0]/avg_prob[1] if avg_prob[1] else 0.0, score_parts[0], score_parts[1], score_parts[1]/score_parts[0]),

    if c==500:
        break

print ''

for t in T:
    print 't =', abs(t), '(w)' if  t < 0 else ''
    print 'pre:', sum(pre[t])/len(pre[t])
    print 'pre_Q:', sum(pre_Q[t])/len(pre_Q[t])
    print 'pre_Q_p:', sum(pre_Q_p[t])/len(pre_Q_p[t])
    print 'pre_pop:', sum(pre_pop[t])/len(pre_pop[t])

    print 'pre_CF:', sum(pre_CF[t])/len(pre_CF[t])
    print ''

print target_user
#f = open('results.txt', 'w')
#for t in T:
#    f.write('t = %d %s\n' % (abs(t), '(w)' if t < 0 else ''))
#    f.write('pre: %f\n' % (sum(pre[t])/len(pre[t]),))
#    f.write('pre_Q: %f\n' % (sum(pre_Q[t])/len(pre_Q[t]),))
#    f.write('\n')
#f.close()


