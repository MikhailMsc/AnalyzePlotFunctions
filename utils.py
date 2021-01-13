import numpy as np


def reorder_palette(palette, order_by):
    indx = np.argsort(order_by)
    indx = {ind: i for i, ind in enumerate(indx)}
    palette = [palette[indx[i]] for i in range(len(palette))]
    return palette

