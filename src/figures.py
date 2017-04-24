'''FIGURES'''
'''taken from http://bkanuka.com/articles/native-latex-plots/'''

import numpy as np
import matplotlib as mpl
mpl.use('pgf')

def size(scale):
    fig_width_pt = 412.56496 # \the\textwidth from latex
    inches_per_pt = 1.0/72.27 # pts to inches
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale # width in inches
    fig_height = fig_width*golden_mean # height in inches
    return [fig_width,fig_height]

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": size(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def new(width,nrows=1,ncols=1):
    f,ax=plt.subplots(nrows=nrows,ncols=ncols)
    return f,ax

def save(path):
    plt.savefig('{}.pgf'.format(path))
    plt.savefig('{}.pdf'.format(path))
