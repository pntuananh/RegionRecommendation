dataset = 'checkin2011'
inputfile = 'FoursquareCheckins20110101-20111231.csv'
    
#dataset = 'SHTiesData'
#inputfile = 'FoursquareCheckins.csv'
    
from collections import defaultdict, Counter
import itertools
import numpy as np

from header import *
import pdb

import networkx as nx

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

import rtree
print 'inserting to Rtree...'
r_tree = rtree.index.Index()
for p in poi_users:
    x,y = coordinates[p]
    r_tree.insert(0, (x,y,x,y), obj=p)

# calculate global user similarity
def similarity(s1,s2):
    return len(s1&s2)*1.0/len(s1|s2)

print 'calculating global user similarity...'
global_user_sim = {}

#c = 0
#for test_user in testing:
#    similar_users = reduce(lambda x,y: x|y, (poi_users[poi] for poi in training[test_user]), set())
#    similar_users.remove(test_user)
#
#    list_sim_user = []
#    for sim_user in similar_users:
#        sim = similarity(training[test_user], training[sim_user])
#        list_sim_user += [(sim, sim_user)]
#
#    list_sim_user.sort(reverse=True)
#    global_user_sim[test_user] = {user:sim for sim, user in list_sim_user[:1000]}#list_sim_user[:100]
#
#    c += 1
#    if c%100 == 0:
#        print '\r%d' % c,
#
#lines = []
#f = open('user_sim.txt', 'w')
#for test_user in global_user_sim:
#    lines += ['%s %s' % (test_user, ' '.join('%s|%s' % (u,s) for u,s in global_user_sim[test_user].iteritems()))]
#
#f.write('\n'.join(lines))
#f.close()
#pdb.set_trace()

f = open('user_sim.txt')
for line in f:
    p = line.strip('\n').split()
    test_user = p[0]
    global_user_sim[test_user] = {}
    for u_s in p[1:]:
        u,s = u_s.split('|')
        global_user_sim[test_user][u] = float(s)
f.close()

T = [1,3,5,10,20,50,-1,-3,-5,-20,-50]
pre = {}
rec = {}
pre_Q = {}
rec_Q = {}
pre_pop = {}
rec_pop = {}

for t in T:
    pre[t] = []
    rec[t] = []

    pre_Q[t] = []
    rec_Q[t] = []

    pre_pop[t] = []
    rec_pop[t] = []


score_parts = [0,0]
avg_prob = (0,0)
decay = 0.1

use_Q = 1
if use_Q:
    print 'computing Q...'
    lines = []
    c = 0
    #for ht, poi_in_towns in hometown_poi.iteritems():
    #    for p in poi_in_towns:
    #        sim_pois = reduce(lambda x,y: x|y, [training[tmp_u] for tmp_u in poi_users[p]], set())
    #        assert p in sim_pois

    #        near_pois = find_near_pois(coordinates[p], 2*R)
    #        assert p in near_pois

    #        sim_pois &= near_pois

    #        user_set_1 = set(u for u in poi_users[p] if user_hometown[u] != poi_hometown[p])
    #        #user_set_1 = poi_users[p]
    #        if not user_set_1:
    #            continue

    #        for sim_p in sim_pois:
    #            user_set_2 = set(u for u in poi_users[sim_p] if user_hometown[u] != poi_hometown[sim_p])
    #            #user_set_2 = poi_users[sim_p]
    #            if not user_set_2:
    #                continue

    #            common = len(user_set_1 & user_set_2)
    #            if common:
    #                #Q[sim_p][p] = common*1.0 / len(user_set_2)
    #                w = common*1.0 / len(user_set_2)
    #                lines.append('%s %s %0.20f' % (sim_p,p,w))


    #        c += 1
    #        if c%1000 == 0:
    #            print '\r%d' % c,
    #            
    ##lines = []
    ##for p in Q:
    ##    for v in Q[p]:
    ##        lines += ['%s %s %0.20f' % (p,v,Q[p][v])]

    #print len(lines)
    #f = open('q_2.txt','w')
    #f.write('\n'.join(lines))
    #f.close()
    #pdb.set_trace()


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

    #f = open('Q_0.010000.txt')
    #for line in f:
    #    p,v,w = line.split()
    #    Q[v][p] = float(w)

    #    c += 1
    #    if c%100000 == 0:
    #        print '\r%d' % c,
    #f.close()

    print ''

    def get_new_score(reg, score):
        global avg_prob
        lamb = 0.5
        new_score = {} #defaultdict(float)
        total_weights = defaultdict(list)

        
        for p in reg:
            #if p not in score:
            #    continue
            sc = score.get(p,0.0)

            for sim_p in Q[p]:
                if sim_p not in reg or sim_p == p:
                    continue

                total_weights[sim_p] += [(Q[p][sim_p], sc)]
                avg_prob = (avg_prob[0] + Q[p][sim_p], avg_prob[1]+1)

                #new_score[sim_p] += lamb * Q[p][sim_p] * score[p]
                #total_weights[sim_p] += Q[p][sim_p]

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


avg_len = []
c = 0
print 'start...'
lines = []
for target_user in testing:
    ot, true_pois = testing[target_user]

    score = defaultdict(float)
    for p in hometown_poi[ot]:
        s = 0
        for user in poi_users[p]:
            s += global_user_sim[target_user].get(user,0)

        if s > 0:
            score[p] = s


    assert len(true_pois) >= 3

    #sample regions
    regions = []
    regions_Q = []
    regions_N = []

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

        new_score = get_new_score(reg, score)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        regions_N += [len(reg)]
        
        if (not true_regions) or (len(cl) == max_n_pos):
            true_regions.add(j)
            max_n_pos = len(cl)

        j += 1
        seen.update(cl)


    for poi in true_pois - seen:
        reg = sample_cover_region([poi])

        sorted_r = sorted([(score.get(p,0),p) for p in reg], reverse=True)
        regions += [sorted_r]

        new_score = get_new_score(reg, score)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]

        regions_N += [len(reg)]

        if (not true_regions):
            true_regions.add(j)
            max_n_pos = 1

        j += 1

    neg_pois = [p for p in hometown_poi[ot] if p not in true_pois]

    np.random.shuffle(neg_pois)

    has_neg_region = 0
    for p in neg_pois:
        reg = find_near_pois(coordinates[p])

        if true_pois & reg:
            continue

        #regions += [sorted([(score[p],p) for p in reg if p in score], reverse=True)]
        regions += [sorted([(score.get(p,0),p) for p in reg], reverse=True)]
        new_score = get_new_score(reg, score)
        regions_Q += [sorted([(new_score.get(p,0),p) for p in reg], reverse=True)]
        regions_N += [len(reg)]
        
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

        if most_pois(regions_N, true_regions):
            pre_pop[t] += [1.0]
        else:
            pre_pop[t] += [0.0]


    c += 1
    if c%10 == 0:
        print '\r%d' % c,
        #print '\r%d %f %f %f %f' % (c, avg_prob[0]/avg_prob[1] if avg_prob[1] else 0.0, score_parts[0], score_parts[1], score_parts[1]/score_parts[0]),

    if c==500:
        break

print ''

#f = open('tests.txt', 'w')
#f.write('\n'.join(lines))
#f.close()
#pdb.set_trace()

for t in T:
    print 't =', abs(t), '(w)' if  t < 0 else ''
    print 'pre:', sum(pre[t])/len(pre[t])
    print 'pre_Q:', sum(pre_Q[t])/len(pre_Q[t])
    print 'pre_pop:', sum(pre_pop[t])/len(pre_pop[t])
    print ''

print target_user
#f = open('results.txt', 'w')
#for t in T:
#    f.write('t = %d %s\n' % (abs(t), '(w)' if t < 0 else ''))
#    f.write('pre: %f\n' % (sum(pre[t])/len(pre[t]),))
#    f.write('pre_Q: %f\n' % (sum(pre_Q[t])/len(pre_Q[t]),))
#    f.write('\n')
#f.close()
