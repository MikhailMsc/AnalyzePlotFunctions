import numpy as np


def reorder_palette(palette, order_by):
    order_colors = sorted([(true_pos, value_pos) for value_pos, true_pos in enumerate(np.argsort(order_by))],
                          key=lambda x: x[0])
    palette = [palette[pos] for _, pos in order_colors]
    return palette
    # indx = np.argsort(order_by)
    # indx = {ind: i for i, ind in enumerate(indx)}
    # palette = [palette[indx[i]] for i in range(len(palette))]
    # return palette


def split_palette(palette, order_by, cnt_1, cnt_2):
    order_colors = sorted([(true_pos, value_pos) for value_pos, true_pos in enumerate(np.argsort(order_by))],
                          key=lambda x: x[0])
    palette_1 = [palette[pos] for _, pos in order_colors[:cnt_1]]
    palette_2 = [palette[pos] for _, pos in order_colors[-cnt_2:]]
    remain_palette = [value for value in palette if value not in palette_1 + palette_2]
    return palette_1, palette_2, remain_palette