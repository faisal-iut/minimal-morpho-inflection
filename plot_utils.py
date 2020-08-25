##Author: https://github.com/seismatica/ngram/blob/6c834edf12aa4b00ad7cc3c8e803c7f3a3cd5e47/analysis/plot_utils.py
import matplotlib.pyplot as plt


def set_style(plt):
    plt.style.use('seaborn')
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 10
    return plt

def plot_em_iteration(avg_ll):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    iterations =  list(avg_ll.keys())
    em_avg_lls = [avg_ll[iteration]['avg_ll'] for iteration in iterations]

    p_lengths = list(range(5))
    p_colors = ['tab:gray', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
    p_labels = ['n3_pscs', 'n2_pscs', 'n1_pscs', 'sc_ps', 'sc']

    em_weight_percents = {}
    for p_length in p_lengths:
        em_weight_percents[p_length] = [avg_ll[iteration]['lambdas'][p_length]*100
                                            for iteration in iterations]


    # ax1.set_xticks(range(11))
    # ax1.set_xlim(0, 10)
    # ax1.set_ylim(-6.6, -6.1)
    ax1.set_ylabel('Average log likelihood')
    ax2.set_ylabel('Interpolation weight (%)')
    ax1.set_xlabel('Iterations')
    ax2.set_xlabel('Iterations')


    for p_length, p_color, p_label in zip(p_lengths, p_colors, p_labels):
        ax2.plot(iterations, em_weight_percents[p_length], color=p_color, marker='o', clip_on=False, label=p_label)

    ax1.plot(iterations, em_avg_lls,color='r', marker='o', clip_on=False)
    ax2.legend(**legend_opts, bbox_to_anchor=(1.04,0), loc='lower left')


legend_opts = {
    'fontsize': 15,
    'frameon': True,
    'framealpha': 1,
    'facecolor': 'white',
    'edgecolor': 'black',
    'labelspacing': 0.1}


def savefig(fig, filename, **kwargs):
    fig.savefig(f'../viz/{filename}', bbox_inches='tight', **kwargs)



plt = set_style(plt)