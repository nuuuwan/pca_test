import copy
import statistics

import numpy as np
from gig import Ent, EntType, GIGTable
from sklearn.decomposition import PCA


def print_line():
    print('-' * 32, end='\n\n')


def roundx(v):
    return round(v, 3)


def roundx_list(vlist):
    return [roundx(v) for v in vlist]


def roundp_list(vlist):
    return [f'{v:.1%}' for v in vlist]


gig_table = GIGTable(
    'government-elections-parliamentary', 'regions-ec', '2015'
)

lk = Ent.from_id('LK')
row_lk = lk.gig(gig_table)
data_lk = {
    k: v
    for k, v in row_lk.dict.items()
    if k not in ['electors', 'polled', 'rejected', 'valid']
}
total_lk = sum(data_lk.values())
fields = [
    x[0] for x in sorted(data_lk.items(), key=lambda x: x[1], reverse=True)
][:10]


print(fields)
print([roundx(data_lk[f] / total_lk) for f in fields])
print_line()

pd_list = Ent.list_from_type(EntType.PD)
name_list = []
v_list = []
for pd in pd_list:
    row = pd.gig(gig_table)
    data = {
        k: v
        for k, v in row.dict.items()
        if k not in ['electors', 'polled', 'rejected', 'valid']
    }

    total = sum(data.values())
    v = [roundx(data[f] / total) for f in fields]
    v_list.append(v)
    name_list.append(f'{pd.id} {pd.name}')
n = len(v_list)
m = len(v_list[0])
print(f'{n=}, {m=}')
print_line()
v_list_orig = copy.deepcopy(v_list)


stats = []
mean_list = []
stdev_list = []
for i in range(0, m):
    m_values = [v[i] for v in v_list]
    mean = statistics.mean(m_values)
    stdev = statistics.stdev(m_values)
    print(i, fields[i], mean, stdev)
    z = [(v - mean) / stdev for v in m_values]
    for h in range(0, n):
        v_list[h][i] = z[h]
    stat = dict(field=fields[i], mean=mean, stdev=stdev)
    stats.append(stat)
    mean_list.append(mean)
    stdev_list.append(stdev)

v_mean = np.array(mean_list).reshape(1, m)
v_stdev = np.array(stdev_list).reshape(1, m)

for v, name in list(zip(v_list, name_list))[:10]:
    print(name.ljust(30), end=' ')
    print(roundx_list(v), end=' ')
    print()
print_line()

n_components = 1
print(f'{n_components=}')
X = np.array(v_list)
print(X.shape)
pca = PCA(n_components=n_components)
pca.fit(X)
print('explained_variance_ratio_', pca.explained_variance_ratio_)
print('singular_values_', pca.singular_values_)
print('mean_', pca.mean_)
print_line()

print('Principle Components')
pc = pca.components_
for x in pc:
    print(roundx_list(x))
print_line()


for v_orig, v, name in list(zip(v_list_orig, v_list, name_list))[:10]:
    print(name)

    t = pca.transform(np.array(v).reshape(1, m))[0]
    print('\t t=', roundx_list(t))

    z = np.matmul(t, pc)
    print('\t z=', z)

    x = z * v_stdev + v_mean
    print('\t x =', roundp_list(v_orig))
    print('\t x^=', roundp_list(x.tolist()[0]))

    print()
print_line()
