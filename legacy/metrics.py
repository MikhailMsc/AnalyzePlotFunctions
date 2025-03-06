import pandas as pd
from matplotlib import pyplot as plt
from config import BASE_CONFIG
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_metrics(data, pred, fact, segments=None, metric='roc', rnd=3, plot_config=None):
    """

    Parameters
    ----------
    data
    pred
    fact
    segments
    metric
    rnd
    plot_config

    Returns
    -------

    """
    base_config = BASE_CONFIG.__copy__()
    if metric == 'pr':
        base_config.title = 'Precision-Recal Curve'
        base_config.xlabel = 'Recall'
        base_config.ylabel = 'Precision'
        base_config.xmax = 1.05
        base_config.line_widths = 2
        base_config.legend_loc = (0.5, 0.65)
    elif metric == 'roc':
        base_config.title = 'ROC-Curve'
        base_config.xlabel = 'False Positive Rate'
        base_config.ylabel = 'True Positive Rate'
        base_config.xmax = 1
        base_config.ymin = -0.02
        base_config.legend_loc = (1, 0.6)
    elif metric == 'lift':
        base_config.title = 'Lift-Chart'
        base_config.xlabel = 'Total Population'
        base_config.ylabel = 'Gain'
        base_config.xmax = 1
        base_config.ymin = -0.02
        base_config.legend_loc = (0.6, 0.65)

    base_config.ymax = 1.05
    base_config.xmin = -0.02
    base_config.legend = True
    base_config.legend_size = 6
    base_config.line_widths = 2
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)

    if type(pred) == str:
        pred = [pred, ]
    if type(fact) == str:
        fact = [fact, ]

    if not segments:
        segments = {'__DELETE__': pd.Series([True] * data.shape[0])}
    data = data[pred + fact]

    lb_template = []
    if len(segments) > 1:
        lb_template.append('{segment_i}')
    if len(pred) > 1:
        lb_template.append('{pred_i}')
    if len(fact) > 1:
        lb_template.append('{target_i}')
    if metric == 'roc':
        lb_template.append('AUC={auc} (Gini={gini})')
    elif metric == 'pr':
        lb_template.append('AUC={auc}')

    lb_template = ', '.join(lb_template)
    min_y = []
    gini = ''
    calced_auc = ''

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        ax = fig.add_subplot(211)
        for y_fact in fact:
            for y_pred in pred:
                for seg in segments:
                    if metric == 'pr':
                        y, x, thresholds = precision_recall_curve(
                            data.loc[segments[seg], y_fact], data.loc[segments[seg], y_pred])
                        calced_auc = round(auc(x, y), rnd)
                    elif metric == 'roc':
                        x, y, thresholds = roc_curve(data.loc[segments[seg], y_fact],
                                                         data.loc[segments[seg], y_pred])
                        calced_auc = round(auc(x, y), rnd)
                        gini = round(2*calced_auc-1, rnd)
                    elif metric == 'lift':
                        tmp = data.loc[segments[seg], :].reset_index(drop=True)
                        gr = tmp.groupby(y_pred)

                        def agg_func(grdf):
                            return pd.Series({'cnt_good': grdf[y_fact].sum(), 'cnt_all': grdf.shape[0]})

                        lift = gr.apply(agg_func)
                        lift.columns = ['cnt_good', 'cnt_all']
                        lift = lift.reset_index().sort_values(y_pred, ascending=False)
                        lift['cnt_good_cum_prc'] = lift['cnt_good'].cumsum() / lift['cnt_good'].sum()
                        lift['cnt_all_cum_prc'] = lift['cnt_all'].cumsum() / lift['cnt_all'].sum()
                        x = lift['cnt_all_cum_prc'].values
                        y = lift['cnt_good_cum_prc'].values
                    else:
                        raise Exception('Wrong metric name!')

                    if lb_template:
                        lb = lb_template.format(segment_i=seg, pred_i=y_pred, target_i=y_fact, auc=calced_auc,
                                                gini=gini)
                    else:
                        lb = None
                    ax.plot(x, y, lw=plot_config.line_widths, label=lb)
                    min_y.append(min(y))

        min_y = min(min_y)
        plot_config.ymin = (min_y - 0.1) if plot_config.ymin is None else plot_config.ymin
        plot_config.appy_configs(fig=fig, ax=ax)
        plt.show()


def calibration(data: 'DataFrame', pred: 'str or list', fact: 'str or list',
                bins: list, segments: dict = {}, return_table=True, plot_config=None):
    """
    Plot calibratin plot for prediction probability.

    Parameters
    ----------
    data
    pred
    act
    bins
    rnd
    segments
    params

    Returns
    -------

    """
    base_config = BASE_CONFIG.__copy__()
    base_config.title = 'Calibration Curve'
    base_config.xlabel = 'Prediction Probability'
    base_config.ylabel = 'Observed Probability'
    base_config.xmax = 1
    base_config.ymax = 1
    base_config.ymin = 0
    base_config.xmin = 0
    base_config.legend = True
    base_config.legend_loc = (1, 0.75)
    base_config.legend_size = 6
    base_config.line_widths = 2
    base_config.color2 = 'black'
    plot_config = base_config if plot_config is None else base_config.merge_other_config(plot_config)

    if type(pred) == str:
        pred = [pred, ]
    if type(fact) == str:
        fact = [fact, ]
    if not segments:
        segments = {'__DELETE__': pd.Series([True] * data.shape[0])}

    labels = ['({}, {}]'.format(float(bins[i - 1]), float(bins[i])) for i in range(1, len(bins))]
    template = pd.DataFrame({'bin': labels, 'bin_num': range(len(labels))})
    lb_template = []
    if len(segments) > 1:
        lb_template.append('{segment_i}')
    if len(pred) > 1:
        lb_template.append('{pred_i}')
    if len(fact) > 1:
        lb_template.append('{target_i}')
    lb_template = ', '.join(lb_template)
    return_table = []

    with plt.style.context(plot_config.style):
        fig = plt.figure(figsize=plot_config.figure_size)
        ax = fig.add_subplot(211)
        ax.plot([0, 1], [0, 1], color=plot_config.color2, linestyle='--', lw=plot_config.line_widths)
        for y_pred in pred:
            data['bin'] = pd.cut(data[y_pred], bins=bins, include_lowest=False)
            data['bin'] = data['bin'].astype('str')
            for y_fact in fact:
                for seg in segments:
                    tmp = data[segments[seg]].groupby('bin').agg(
                        {y_pred: ['size', 'mean'], y_fact: 'mean'}).reset_index()
                    tmp.columns = ['bin', 'cnt', 'pred_prob', 'obs_prob']
                    tmp['population'] = round(100 * tmp['cnt'] / tmp['cnt'].sum(), 2)
                    if lb_template:
                        lb = lb_template.format(segment_i=seg, pred_i=y_pred, target_i=y_fact)
                    else:
                        lb = None
                    ax.plot(tmp['pred_prob'], tmp['obs_prob'], 'o-', lw=plot_config.line_widths, label=lb)
                    tmp = template.merge(tmp, on='bin', how='left')
                    tmp['segment'] = seg
                    tmp['pred'] = y_pred
                    tmp['fact'] = y_fact
                    return_table.append(tmp)

        plot_config.appy_configs(fig=fig, ax=ax)
        plt.show()
        if return_table:
            return_table = pd.concat(return_table, axis=0, sort=False)
            return_table['bin'] = pd.Categorical(return_table['bin'], categories=labels, ordered=True)
            return_table.sort_values(['bin', 'segment', 'pred', 'fact'], ascending=True)
            if len(segments) == 1:
                del return_table['segment']
            if len(fact) == 1:
                del return_table['fact']
            if len(pred) == 1:
                del return_table['pred']

            return return_table
