from functools import cached_property

import numpy as np
from gig import Ent, EntType, GIGTable
from utils import Log

log = Log('Election')


class Election:
    IGNORE_KEYS = ['electors', 'polled', 'rejected', 'valid']

    def __init__(self, election_type, year):
        self.election_type = election_type
        self.year = year

    @cached_property
    def gig_table(self):
        return GIGTable(
            f'government-elections-{self.election_type}',
            'regions-ec',
            f'{self.year}',
        )

    @staticmethod
    def get_norm_data_from_row(row, ordered_data_keys=None):
        should_build_keys = ordered_data_keys is None

        unnorm_data = {
            k: v for k, v in row.dict.items() if k not in Election.IGNORE_KEYS
        }
        total = sum(unnorm_data.values())

        if should_build_keys:
            ordered_data_keys = [
                x[0]
                for x in sorted(
                    unnorm_data.items(), key=lambda x: x[1], reverse=True
                )
            ][:10]
            log.debug(f'{ordered_data_keys=}')

        norm_data = {
            k: unnorm_data.get(k, 0) / total for k in ordered_data_keys
        }
        non_other_total = sum(norm_data.values())
        norm_data['_other'] = 1 - non_other_total

        if not should_build_keys:
            return norm_data

        non_other_data = {k: v for k, v in norm_data.items() if v > 0.025}
        p_non_other = sum(non_other_data.values())
        p_other = 1 - p_non_other
        non_other_data['_other'] = p_other
        return non_other_data

    @cached_property
    def lk_result(self):
        lk = Ent.from_id('LK')
        row_lk = lk.gig(self.gig_table)
        return Election.get_norm_data_from_row(row_lk)

    @cached_property
    def ordered_data_keys(self):
        return list(self.lk_result.keys())

    @cached_property
    def pd_results_idx(self):
        pd_list = Ent.list_from_type(EntType.PD)
        idx = {}
        for pd in pd_list:
            row = pd.gig(self.gig_table)
            data = Election.get_norm_data_from_row(
                row, self.ordered_data_keys
            )
            pd_key = f'{pd.id} {pd.name}'
            idx[pd_key] = data
        return idx

    @staticmethod
    def get_history(election_list):
        idx = {}
        for election in election_list:
            pd_results_idx = election.pd_results_idx
            for pd_id, party_to_p in pd_results_idx.items():
                if pd_id not in idx:
                    idx[pd_id] = {}
                for party, p in party_to_p.items():
                    key = f'{election.year[-2:]}{party}'
                    idx[pd_id][key] = p
        keys = list(list(idx.values())[0].keys())
        ordered_idx = {k: {k1: v[k1] for k1 in keys} for k, v in idx.items()}

        np_vectors = np.array(
            list([[v1 for v1 in v.values()] for v in ordered_idx.values()])
        )

        n = len(np_vectors)
        m = len(np_vectors[0])
        mean_list = []
        stdev_list = []
        for i in range(0, m):
            m_values = [v[i] for v in np_vectors]
            mean = np.mean(m_values)
            stdev = np.std(m_values)
            z = [(v - mean) / stdev for v in m_values]
            for h in range(0, n):
                np_vectors[h][i] = z[h]
            mean_list.append(mean)
            stdev_list.append(stdev)

        np_mean = np.array(mean_list).reshape(1, m)
        np_stdev = np.array(stdev_list).reshape(1, m)
        log.debug(f'{np_mean=}')
        log.debug(f'{np_stdev=}')

        pd_ids = list(ordered_idx.keys())
        return ordered_idx, pd_ids, keys, np_vectors, np_mean, np_stdev

    @staticmethod
    def format(np_vector):
        return ''.join([f'{v:.2f}'.rjust(6) for v in np_vector])

def main():
    (
        ordered_idx,
        pd_ids,
        keys,
        np_vectors,
        np_mean,
        np_stdev,
    ) = Election.get_history(
        [Election('parliamentary', '2020'), Election('parliamentary', '2015'),Election('parliamentary', '2010'), Election('parliamentary', '2004')]
    )
    

    print('-' * 32)
    print('PD'.ljust(30), ''.join([f'{k}'.rjust(6) for k in keys]))
    for i, [pd_id, np_vector] in enumerate(list(zip(pd_ids, np_vectors))):
        if i % 10 != 0:
            continue
        print(pd_id.ljust(30), Election.format(np_vector))


    from sklearn.decomposition import PCA
    n_components= 3
    print(f'{n_components=}')
    pca = PCA(n_components=n_components)
    pca.fit(np_vectors)
    pc_list = pca.components_
    
    print('-' * 32)
    print('PD'.ljust(30), ''.join([f'{k}'.rjust(6) for k in keys]))
    for i, pc in enumerate(pc_list):
        print(f'pc{i+1}'.ljust(30), Election.format(pc))


if __name__ == "__main__":
    main()
