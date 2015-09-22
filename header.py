from math import radians, cos, sin, asin, sqrt
from collections import Counter

def haversine(point1, point2):
    lon1, lat1 = point1
    lon2, lat2 = point2
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 

    # 6367 km is the radius of the Earth
    km = 6367 * c
    return km 

def my_open(fname,sep=None):
    for line in open(fname):
        yield line.strip('\n').split(sep) 

def my_open_ex(fnames,sep=' '):
    for fname in fnames:
        for line in open(fname):
            yield line.strip('\n').split(sep) 

def write_to_file(filename, lines):
    f = open(filename, 'w')
    f.write('\n'.join(lines))
    f.close()

def show_avg(l):
    if not l: return 0.0
    print sum(l)*1.0/len(l)

def show_histogram(l,t=10):
    c = Counter(l)
    s = sum(c.itervalues())*1.0
    a = 0
    for k in sorted(c)[:t]:
        a += c[k]
        print k, c[k], c[k]/s, a/s

