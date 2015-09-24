from header import * 
from collections import defaultdict
import numpy as np

import rtree

import warnings
warnings.filterwarnings('error')

import pdb

K = 50
R = 5
N = 50
regular = 5.0
q_regular = 0

reload = 0
 
lr = 0.01
alpha1 = 0.2
alpha2 = 0.5
lamb = 0.5

delta_lat = 0.009 
delta_lon = 0.012 
R_lat = delta_lat*R
R_lon = delta_lon*R

print 'reading training file...'
user_pois = {}
poi_users = defaultdict(set)

for parts in my_open('training.txt'):
    user = parts[0]
    pois = parts[1:]

    user_pois[user] = set(pois)
    for p in pois:
        poi_users[p].add(user)

print 'reading locations...'
coordinates = {}
for parts in my_open('locations.txt'):
    poi, lon, lat = parts
    coordinates[poi] = (float(lon),float(lat))

def distance(p1,p2):
    return haversine(coordinates[p1], coordinates[p2])

print 'inserting to RTree...'
r_tree = rtree.index.Index()
for poi in poi_users:
    x,y = coordinates[poi]
    r_tree.insert(0, (x,y,x,y), obj=poi)


print 'reading hometown...'
user_hometown = {}
hometown_user = defaultdict(list)
dataset = 'checkin2011'
f = open(dataset + '\\hometown.csv' )
f.readline()

for line in f:
    user, hometown = line.strip('\n').split(',')

    user_hometown[user] = hometown
    hometown_user[hometown].append(user)

poi_hometown = {}
hometown_poi = {}

for parts in my_open('hometown_poi.txt'):
    hometown = parts[0]
    pois = parts[1:]
    hometown_poi[hometown] = pois
    for p in pois:
        poi_hometown[p] = hometown


training_set = []
for user, true_pois in user_pois.iteritems():
    town_pois = defaultdict(list)
    for poi in true_pois:
        town = poi_hometown[poi]
        town_pois[town] += [poi]

    training_set += [(user,t,pois) for t,pois in town_pois.iteritems()]


def load_Q():
    print 'loading Q...'
    l = []
    Q = defaultdict(dict)
    prev = None
    c = 0
    if q_regular:
        for line in my_open('Q_%f.txt' % q_regular):
            p1, p2, w = line
            # p2 --> p1

            if prev is not None and p1 != prev:
                l.sort(reverse=True)
                for w, p in l[:50]:
                    Q[prev][p] = w

                l = []

            l += [(float(w),p2)]
            prev = p1

            c += 1
            if c%100000 == 0:
                print '\r%d' %c,
        print ''

        if l:
            l.sort(reverse=True)
            for w, p in l[:50]:
                Q[prev][p] = w
    else:
        for line in my_open('q_2.txt'):
            p,v,w = line
            # p --> v
            Q[v][p] = float(w)

            c += 1
            if c%100000 == 0:
                print '\r%d' % c,
        print ''

    return Q

Q = load_Q()

def sample_a_region(x,y):
    min_x = x - R_lon
    min_y = y - R_lat

    x_rand = min_x + R_lon*np.random.random()
    y_rand = min_y + R_lat*np.random.random()

    return set(item.object for item in r_tree.intersection((x_rand,y_rand,x_rand+R_lon,y_rand+R_lat), objects=True))


def sample_pos_region(pos_pois):
    pos = np.random.choice(pos_pois)
    x,y = coordinates[pos]

    pos_reg = sample_a_region(x,y)

    return pos_reg, [p for p in pos_pois if p in pos_reg]


def sample_neg_region(ot, pos_pois, max_n_pos):
    np.random.shuffle(hometown_poi[ot])

    for neg in hometown_poi[ot]:
        if neg in pos_pois: continue

        x,y = coordinates[neg]
        neg_reg = sample_a_region(x,y)

        pos_in_reg = [p for p in pos_pois if p in neg_reg]
        if len(pos_in_reg) < max_n_pos:
            return neg_reg, pos_in_reg

    return None, None
        

def get_region_score(user, pois):
    scores = {}
    #pois = list(pois)
    for p in pois:
        scores[p] = (user_pref[user] + user_pref_d[user]).dot(poi_pref[p])

    return scores


def get_region_q_score(scores,top_pois):
    q_scores = {p:0.0 for p in scores}
    local_weights = {}
    for p in scores:
        local_weights[p] = {p1:w for p1,w in Q[p].iteritems() if p1 in top_pois}

        if local_weights[p]:
            s_w = sum(local_weights[p].itervalues())
            for p1 in local_weights[p]:
                local_weights[p][p1] /= s_w

        for p1,w in local_weights[p].iteritems():
            q_scores[p] += w * scores[p1] * lamb

        q_scores[p] += scores[p] * (1-lamb)

    sorted_pois = sorted([p for p in q_scores], key=lambda x:q_scores[x], reverse=True)
    weights = {p:1./(i+1) for i,p in enumerate(sorted_pois[:N])}
    total_weights = sum(weights.itervalues())
    for p in weights:
        weights[p] /= total_weights

    return q_scores, weights, local_weights


def user_derivative_1(pos_reg, neg_reg, pos_r_weights, neg_r_weights, 
        local_p_weights, local_n_weights):

    def compute_part(reg, weights, local_weights):
        v = np.zeros(K)
        for p,w in weights.iteritems():
            temp = poi_pref[p] * (1-lamb)
            for p1,w1 in local_weights[p].iteritems():
                temp += w1*poi_pref[p1] * lamb
                
            v += temp*w

        return v

    pos_part = compute_part(pos_reg, pos_r_weights, local_p_weights)
    neg_part = compute_part(neg_reg, neg_r_weights, local_n_weights)

    return pos_part - neg_part


def pois_derivative_1(pos_reg, neg_reg, pos_r_weights, neg_r_weights, user,
        local_p_weights, local_n_weights):

    def compute_part(poi, weights, local_weights):
        ret = weights.get(poi,0) * (1-lamb)
        for p,w in weights.iteritems():
            if poi not in local_weights[p]: continue
            ret += w * local_weights[p][poi] * lamb

        return ret

    der = {}
    all_pois = pos_reg | neg_reg
    for p in all_pois:
        part1 = part2 = 0
        if p in pos_reg:
            part1 = compute_part(p, pos_r_weights, local_p_weights)
        if p in neg_reg:
            part2 = compute_part(p, neg_r_weights, local_n_weights)

        der[p] = (part1-part2)*(user_pref[user] + user_pref_d[user])

    return der


def update_pref(pref, amount, regular):
    pref += amount 
    L2 = np.linalg.norm(pref,2)
    if L2 > regular:
        pref *= regular/L2

#def update_pref(pref, factor, amount, regular):
#    pref += factor * (amount - regular*pref)

#training
if reload:
    print 'reloading...'

    user_pref = {}
    poi_pref = {}

    for p in my_open('model\\user_factors_%f_%f.txt' % (regular, q_regular)):
        user = p[0]
        user_pref[user] = np.array(map(np.float64,p[1:]))

    for p in my_open('model\\poi_factors_%f_%f.txt' % (regular, q_regular)):
        poi = p[0]
        poi_pref[poi] = np.array(map(np.float64,p[1:]))
else:
    user_pref   = defaultdict(lambda: np.random.random(K)*2 - 1)
    user_pref_d = defaultdict(lambda: np.random.random(K)*2 - 1)
    poi_pref    = defaultdict(lambda: np.random.random(K)*2 - 1)

print 'Traning size:', len(training_set)
for it in range(50):
    print 'iteration #%d' % it
    np.random.shuffle(training_set)

    count = 0
    for user, town, pois in training_set:
        count += 1
        if count%100==0:
            print '\rtraining instance #%d' % (count,),

        is_hometown = town == user_hometown[user]
        pos_reg, pos_in_p_reg = sample_pos_region(pois)
        neg_reg, pos_in_n_reg = sample_neg_region(town,pois,len(pos_in_p_reg))

        ####rank regions
        if not is_hometown:
            if neg_reg is not None:
                top_pois_pos_reg = set(sorted([p for p in pos_reg], key=lambda x:len(poi_users[x]),reverse=True)[:5])
                top_pois_neg_reg = set(sorted([p for p in neg_reg], key=lambda x:len(poi_users[x]),reverse=True)[:5])

                pos_r_scores = get_region_score(user,pos_reg)
                pos_q_r_scores, pos_r_weights, local_p_weights = get_region_q_score(pos_r_scores,top_pois_pos_reg)

                neg_r_scores = get_region_score(user,neg_reg)
                neg_q_r_scores, neg_r_weights, local_n_weights = get_region_q_score(neg_r_scores,top_pois_neg_reg)

                score_of_pos_reg = sum(w*pos_q_r_scores[p] for p,w in pos_r_weights.iteritems())
                score_of_neg_reg = sum(w*neg_q_r_scores[p] for p,w in neg_r_weights.iteritems())
                try:
                    der = 1/(1+np.exp(score_of_pos_reg-score_of_neg_reg))
                except:
                    print score_of_pos_reg, score_of_neg_reg
                    der = 1.0 if score_of_pos_reg < score_of_neg_reg else 0.0

                pos_reg_tmp = set(pos_r_scores.keys())
                neg_reg_tmp = set(neg_r_scores.keys())

                der_user = user_derivative_1(pos_reg_tmp, neg_reg_tmp, pos_r_weights, neg_r_weights,
                        local_p_weights, local_n_weights)
                update_pref(user_pref[user], alpha1*lr*der*der_user, regular)
                update_pref(user_pref_d[user], alpha1*lr*der*der_user, regular)

                #update_pref(user_pref[user], alpha1*lr, der*der_user, regular)
                #update_pref(user_pref_d[user], alpha1*lr, der*der_user, regular)
           
                #update pois
                #print 'update pois...'
                der_pois = pois_derivative_1(pos_reg_tmp, neg_reg_tmp, pos_r_weights, neg_r_weights, user,
                        local_p_weights, local_n_weights)
                for p,d in der_pois.iteritems():
                    update_pref(poi_pref[p], alpha1*lr*der*d, regular)
                    #update_pref(poi_pref[p], alpha1*lr, der*d, regular)


        ####rank pois in region
        #print '\r%d RANK POIS...' % count,
        Ds = []
        neg_pois = [p for p in pos_reg if p not in pos_in_p_reg]
        for pos in pos_in_p_reg:
            sample_n_pois = np.random.choice(neg_pois, size=10, replace=False) if len(neg_pois) > 10 else neg_pois
            for neg in sample_n_pois:
                Ds += [(pos,neg)]

        np.random.shuffle(Ds)
        for pos, neg in Ds:
            try:
                u_pref = user_pref[user].copy()
                if not is_hometown:
                    u_pref += user_pref_d[user]

                pos_score = u_pref.dot(poi_pref[pos])
                neg_score = u_pref.dot(poi_pref[neg])

                der = 1/(1+np.exp(pos_score - neg_score))
            except:
                print pos_score, neg_score

            #update user
            der_user = poi_pref[pos] - poi_pref[neg] #user_derivative_2(pos_reg, pos, neg)
            update_pref(user_pref[user], alpha2*lr*der*der_user, regular)
            #update_pref(user_pref[user], alpha2*lr, der*der_user, regular)
            if not is_hometown:
                update_pref(user_pref_d[user], alpha2*lr*der*der_user, regular)
                #update_pref(user_pref_d[user], alpha2*lr, der*der_user, regular)

            #update pois
            update_pref(poi_pref[pos], alpha2*lr*der*user_pref[user], regular)
            update_pref(poi_pref[neg], -alpha2*lr*der*user_pref[user], regular)
            #update_pref(poi_pref[pos], alpha2*lr, der*u_pref, regular)
            #update_pref(poi_pref[neg], alpha2*lr, -der*u_pref, regular)

    print ''

    #save to file
    print 'save to file...'
    lines = []
    for user, factors in user_pref.iteritems():
        lines.append('%s %s' % (user, ' '.join('%f' % f for f in factors)))
    write_to_file('model\\user_factors_%f_1.txt' % (regular, ), lines)

    lines = []
    for user, factors in user_pref_d.iteritems():
        lines.append('%s %s' % (user, ' '.join('%f' % f for f in factors)))
    write_to_file('model\\user_factors_d_%f_1.txt' % (regular, ), lines)

    lines = []
    for poi, factors in poi_pref.iteritems():
        lines.append('%s %s' % (poi, ' '.join('%f' % f for f in factors)))
    write_to_file('model\\poi_factors_%f_1.txt' % (regular, ), lines)

