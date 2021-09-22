# distutils: language = c++
# distutils: include_dirs = NP_INCLUDE

import numpy as np
cimport numpy as np

from libcpp.deque cimport deque
from libcpp.pair cimport pair

def coast_mask(np.ndarray[np.int_t, ndim=2] ocn_mask):
    cst_mask = 0 * ocn_mask
    # land to the west
    cst_mask[(ocn_mask > 0) & (np.roll(ocn_mask, 1, axis=1) == 0)] = 1
    # land to the east
    cst_mask[(ocn_mask > 0) & (np.roll(ocn_mask, -1, axis=1) == 0)] = 1
    # land to the south
    cst_mask[(ocn_mask > 0) & (np.roll(ocn_mask, 1, axis=0) == 0)] = 1

    # land to the north, handling tripole
    nom = np.roll(ocn_mask, -1, axis=0)
    nom[-1,:] = ocn_mask[-1, ::-1]
    cst_mask[(ocn_mask > 0) & (nom == 0)] = 1

    return cst_mask

def nearest_coastal_cell(np.ndarray[np.int_t, ndim=2] ocn_id, np.ndarray[np.int_t, ndim=2] cst_mask):
    cdef deque[pair[np.int_t, np.int_t]] stack

    cst_nrst_ocn_id = np.where(cst_mask, ocn_id, -1)
    ocn_nj = ocn_id.shape[0]
    ocn_ni = ocn_id.shape[1]

    done = 0 * cst_mask
    seen = 1 * cst_mask

    cst_points = np.where(cst_mask)
    for p in zip(*cst_points):
        stack.push_back(p)

    processed = 0
    to_process = stack.size()
    direction = 0

    while not stack.empty():
        if processed == to_process:
            to_process = stack.size()
            processed = 0
            direction = (direction + 1) % 4

        p = stack.front()
        stack.pop_front()

        processed += 1

        if done[p] == 4: continue
        done[p] += 1

        pn = [
            (p[0], (p[1] - 1 + ocn_ni) % ocn_ni),
            (p[0], (p[1] + 1) % ocn_ni),
            (min(p[0] + 1, ocn_nj - 1), p[1]),
            (max(p[0] - 1, 0), p[1]),
        ][direction]

        stack.push_back(p)
        if not seen[pn]:
            stack.push_back(pn)
            cst_nrst_ocn_id[pn] = cst_nrst_ocn_id[p]
            seen[pn] = 1

    return cst_nrst_ocn_id
