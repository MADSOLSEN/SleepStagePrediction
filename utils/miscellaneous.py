
def create_pairs(list_for_indexing):
    N = len(list_for_indexing)
    if N == 1:
        return []
    else:
        pair_list = []
        for n0 in range(N):
            for n1 in range(n0 + 1, N):
                pair_list += [(list_for_indexing[n0], list_for_indexing[n1])]

    return pair_list

