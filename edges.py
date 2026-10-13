"""The model's edge, as the pictures treat it (SeaWise radar build 7, item 14).

Pure numpy, so `test_render.py` can pin it without eccodes. Two helpers:

`edge_weight` fades a painted layer over the last grid cells of the model's
coverage instead of cutting it. The cloud layer drew full cloud right up to
the model's southern edge and nothing past it, and on the national picture's
5 km pixels that cut was a staircase across the Gulf (B-262). Reflectivity is
never faded: rain at the model's edge keeps its colour, and outside the model
stays unpainted, as before.

`inside_mask` says which samples of a field lie inside the model's grid. The
wind renderer sampled with `mode="nearest"` and no mask, so south of the
model's edge the picture carried the edge row on, smeared over open water
where the radar and cloud draw nothing (B-271). The mask goes into the wind
picture's unused blue channel: 255 inside, 0 outside. Installed builds ignore
that channel; the new build reads 0 as "no field here".
"""
import numpy as np


def edge_weight(fx, fy, nx, ny, cells):
    """1 inside the model, falling to 0 over its last `cells` grid cells.

    `fx`, `fy` are grid coordinates (cells, fractional), `nx` by `ny` the
    grid; a point outside the grid weighs 0. With `cells` 0 or None the
    weight is 1 inside and 0 outside: the cut as before.
    """
    fx = np.asarray(fx, dtype=float)
    fy = np.asarray(fy, dtype=float)
    inside = (fx >= 0) & (fx < nx - 1) & (fy >= 0) & (fy < ny - 1)
    if not cells:
        return inside.astype(float)
    distance = np.minimum(np.minimum(fx, nx - 1 - fx), np.minimum(fy, ny - 1 - fy))
    return np.where(inside, np.clip(distance / float(cells), 0.0, 1.0), 0.0)


def inside_mask(col, row, nx, ny):
    """True where a sample (grid column, row) lies inside an `nx` by `ny` grid."""
    col = np.asarray(col, dtype=float)
    row = np.asarray(row, dtype=float)
    return (col >= 0) & (col <= nx - 1) & (row >= 0) & (row <= ny - 1)
