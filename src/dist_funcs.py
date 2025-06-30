import torch


def max_normalized_euclidean_distance(repr0, repr1, params):
    repr0 = torch.FloatTensor(repr0)
    repr1 = torch.FloatTensor(repr1)
    dist0 = torch.cdist(repr0, repr0, p=2)
    dist1 = torch.cdist(repr1, repr1, p=2)
    dist0 = dist0 / dist0.max()
    dist1 = dist1 / dist1.max()
    sim0_2d = 1 - dist0
    sim1_2d = 1 - dist1

    return dict(sim0=sim0_2d, sim1=sim1_2d, dist0=dist0, dist1=dist1)


def zp_local_scaling_euclidean_distance(repr0, repr1, params):
    repr0 = torch.FloatTensor(repr0)
    repr1 = torch.FloatTensor(repr1)
    scaling_neighbor = params['scaling_neighbor']
    beta = params['beta']
    normalize_diff_mat_by_abs_max = params.get('normalize_diff_mat_by_abs_max', False)
    dist0 = torch.cdist(repr0, repr0, p=2)
    dist1 = torch.cdist(repr1, repr1, p=2)
    r0_sort = dist0.argsort()
    r0_nn = torch.argsort(r0_sort)
    r1_sort = dist1.argsort()
    r1_nn = torch.argsort(r1_sort)

    ls0 = torch.gather(dist0, 1, r0_sort[:, None, scaling_neighbor])
    ls1 = torch.gather(dist1, 1, r1_sort[:, None, scaling_neighbor])

    ls0_denom = ls0 @ ls0.T
    ls1_denom = ls1 @ ls1.T

    dist0 = (dist0 * dist0.T) / ls0_denom
    dist1 = (dist1 * dist0.T) / ls1_denom
    # dist0 = dist0 / dist0.max()
    # dist1 = dist1 / dist1.max()

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(1, 2)
    # axes[0].hist(dist0.flatten().cpu().numpy(), bins=100)
    # axes[1].hist(dist1.flatten().cpu().numpy(), bins=100)
    # plt.show()
    sim0_2d = torch.exp(-beta * dist0)
    sim1_2d = torch.exp(-beta * dist1)

    return dict(sim0=sim0_2d, sim1=sim1_2d, dist0=dist0, dist1=dist1)


def scale_invariant_local_biased_distance(repr0, repr1, params):
    beta = params.get('beta', 5)
    num_samples = repr0.shape[0]
    gamma = params.get('gamma', None)
    if gamma is None:
        gamma_scale = params.get('gamma_scale', 80)
        gamma = gamma_scale / num_samples

    print(f"Beta: {beta}, Gamma: {gamma}")

    repr0 = torch.FloatTensor(repr0)
    repr1 = torch.FloatTensor(repr1)

    r0_dm = torch.cdist(repr0, repr0)
    r1_dm = torch.cdist(repr1, repr1)

    r0_sort = r0_dm.argsort()
    r0_dm = torch.argsort(r0_sort).float() + 1
    r1_sort = r1_dm.argsort()
    r1_dm = torch.argsort(r1_sort).float() + 1

    sim0_2d = 1 - r0_dm / r0_dm.max()
    sim1_2d = 1 - r1_dm / r1_dm.max()

    return dict(sim0=sim0_2d, sim1=sim1_2d, dist0=r0_dm, dist1=r1_dm)
