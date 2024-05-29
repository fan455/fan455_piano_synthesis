# fast plot by wrapping matplotlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
#plt.rcParams['figure.dpi'] = 150
#plt.rcParams['savefig.dpi'] = 150
#plt.rcParams['font.family'] ='sans-serif'
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams.update({'font.size': 11.5})

# The default blue color is 'tab:blue' or '#1f77b4'

def plot(y, x=None, yaxis=1, title='title', xlabel='x', ylabel='y', xtick=None, ytick=None, \
         xticklabel=None, yticklabel=None, xtick_kwargs=dict(labelsize=9, labelrotation=45), \
         ytick_kwargs=None, grid=True, bgcolor='#d1ddc5', **kwargs):
    # x is 1d array, y can be 1d or 2d array.
    # If y is 2d, please notice that by default the second axis of y is plotted.    
    if y.ndim == 2:
        if yaxis == 1:
            y = np.swapaxes(y, 0, 1)        
    if x is None:
        x = np.arange(y.shape[0])

    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    
    ax.plot(x, y, **kwargs)
    
    if xtick is not None:
        ax.set_xticks(xtick)
        if xticklabel is not None:
            ax.set_xticklabels(xticklabel)
        if xtick_kwargs is None:
            xtick_kwargs = {}
        ax.tick_params(axis='x', **xtick_kwargs)

    if ytick is not None:
        ax.set_yticks(ytick)
        if yticklabel is not None:
            ax.set_yticklabels(yticklabel)
        if ytick_kwargs is None:
            ytick_kwargs = {}
        ax.tick_params(axis='y', **ytick_kwargs)
        
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
        
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_multi(y, x=None, yaxis=1, linelabels=None, legendkwargs=None, \
               title='title', xlabel='x', ylabel='y', xtick=None, ytick=None, \
               xticklabel=None, yticklabel=None, \
               xtick_kwargs=dict(labelsize=9, labelrotation=45), ytick_kwargs=None, \
               grid=True, bgcolor='#d1ddc5', kwargslist=None, **kwargs):
    # Plot multiple lines in the same graph.
    # x is a 1d array with shape (npoints,).
    # y is a 2d array with shape (N, npoints) if yaxis=1, or shape (npoints, N) if yaxis=0.
    # kwargslist is a list containing N dictionaries with kwargs sent to ax.plot() corresponding to the N lines.
    # Please notice that if both kwargslist and **kwargs is not None, they should not have any intersection.
    # linelabels is a list containing N strings setting the labels of lines.
    # legendkwargs is a dictionary of legend parameters, sent to ax.legend()
    if type(y) == list:
        y = np.array(y)
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    N = y.shape[0]        
    if x is None:
        x = np.arange(y.shape[1])        
    if linelabels is None:
        linelabels = list(f'y{i}' for i in range(N))        
    if kwargslist is None:
        kwargslist = [{}] * N
        
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    
    for i in range(N):
        ax.plot(x, y[i, :], **kwargslist[i], **kwargs)
            
    if legendkwargs is None:
        legendkwargs = {}
    ax.legend(linelabels, **legendkwargs)

    if xtick is not None:
        ax.set_xticks(xtick)
        if xticklabel is not None:
            ax.set_xticklabels(xticklabel)
        if xtick_kwargs is None:
            xtick_kwargs = {}
        ax.tick_params(axis='x', **xtick_kwargs)

    if ytick is not None:
        ax.set_yticks(ytick)
        if yticklabel is not None:
            ax.set_yticklabels(yticklabel)
        if ytick_kwargs is None:
            ytick_kwargs = {}
        ax.tick_params(axis='y', **ytick_kwargs)
        
    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
        
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')    
    plt.show()
    
def subplots(y, x=None, nrows=None, ncols=1, yaxis=1, title=None, subtitle=None, \
             xlabel='x', ylabel='y', grid=False, bgcolor='#d1ddc5', kwargslist=None, **kwargs):
    # x is a 1d array, y is a 2d array with shape(n_subplots, x.size).
    if type(y) == list:
        y = np.array(y)
    if yaxis == 0:
        y = np.swapaxes(y, 0, 1)
    N = y.shape[0]    
    if nrows is None:
        nrows = y.shape[0]        
    if x is None:
        x = np.arange(y.shape[1])
    if kwargslist is None:
        kwargslist = [{}] * N
        
    fig, ax = plt.subplots(nrows, ncols, facecolor=bgcolor)
    ax = ax.reshape(ax.size)
        
    if subtitle:
        if isinstance(subtitle, str):
            for i in range(N):
                ax[i].set_title(f'{subtitle} {i+1}', fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargslist[i], **kwargs)
        elif isinstance(subtitle, (list, tuple)):
            assert len(subtitle) == N
            for i in range(N):
                ax[i].set_title(subtitle[i], fontsize='medium')     
                ax[i].set_facecolor(bgcolor)
                ax[i].plot(x, y[i, :], **kwargslist[i], **kwargs)
        else:
            raise ValueError('Your subtitle type {type(subtitle)} is not supported.')
    else:
        for i in range(N):
            ax[i].set_facecolor(bgcolor)
            ax[i].plot(x, y[i, :], **kwargslist[i], **kwargs)

    nEmpty = nrows * ncols - N
    if nEmpty > 0:
        for i in range(nEmpty):
            ax[-nEmpty].set_facecolor(bgcolor)            
    if grid:
        for i in range(N):
            ax[i].grid(color='grey', linewidth='0.75', linestyle='-.')            
    if title:
        fig.suptitle(title)
        
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_pdf(x, title='probability density function', xlabel='x', ylabel='density', bgcolor='#D1DDC5', grid=True):
    """
    Plot the probability density function of x.
    x: 1d array.
    """
    sns.set(rc={'axes.facecolor':bgcolor, 'axes.edgecolor':'grey', 'figure.facecolor':bgcolor})
    ax = sns.kdeplot(data=x)

    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')

    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_scatter(y, x, title='scatter plot', xlabel='x', ylabel='y', bgcolor='#D1DDC5', grid=True):
    """
    Scatter plot of x and y.
    y, x: 1d array.
    """
    sns.set(rc={'axes.facecolor':bgcolor, 'axes.edgecolor':'grey', 'figure.facecolor':bgcolor})
    ax = sns.scatterplot(x=x, y=y)

    if grid:
        ax.grid(color='grey', linewidth='1', linestyle='-.')

    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()    

def plot_itp(y, x, points, title='title', xlabel='x', ylabel='y', \
             grid=True, bgcolor='#d1ddc5', \
             pointskwargs=dict(linestyle='', marker='.', markersize=9, mec='black', mfc='black'), \
             **kwargs):
    # Plot for interpolation.
    # points is a 2d array with shape(n_points, 2), that will be annotated.
    # points = np.array([[x1, y1], [x2, y2],..., [xn, yn]])
    points = np.array(points)
    n_points = points.shape[0]
    px, py = points[:, 0], points[:, 1]

    if pointskwargs is None:
        pointskwargs = {}
        
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)    
    ax.plot(x, y, **kwargs)
    ax.plot(px, py, **pointskwargs, **kwargs)

    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
        
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_scale(y, x, xscale='log', yscale=None, xscale_kwargs=None, yscale_kwargs=None, \
               title='title', xlabel='x', ylabel='y', grid=True, bgcolor='#d1ddc5', **kwargs):
    # Plot with x and/or y axis scaled. By default x is scaled by "log10".
    # The types of xscale_kwargs and yscale_kwargs should be "dict", so use the "dict()" function.
    # Please refer to "matplotlib.scale".
    # xscale or yscale: 'asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog'
    fig, ax = plt.subplots(facecolor=bgcolor)
    ax.set_facecolor(bgcolor)
    
    if xscale is not None:
        if xscale_kwargs is None:
            xscale_kwargs = {}
        ax.set_xscale(xscale, **xscale_kwargs)
    if yscale is not None:
        if yscale_kwargs is None:
            yscale_kwargs = {}
        ax.set_yscale(yscale, **yscale_kwargs)
    ax.plot(x, y, **kwargs)

    if grid:
        ax.grid(color='grey', linewidth='0.75', linestyle='-.')
        
    plt.title(title)
    plt.xlabel(xlabel, loc='right')
    plt.ylabel(ylabel, loc='center')
    plt.show()

def plot_au_mono(au, sr, title='title', grid=True, bgcolor='#D1DDC5', **kwargs):
    """
    Plot mono audio array.
    au: 1d audio array of shape (nsamples,).
    sr: sample rate.
    """
    t = np.arange(au.shape[0])/sr
    plot(y, t, title=title, xlabel='time (s)', ylabel='amplitude', grid=grid, \
         bgcolor=bgcolor, **kwargs)

def plot_au_stereo(au, sr, title='title', grid=True, bgcolor='#D1DDC5', **kwargs):
    """
    Plot stereo audio array.
    au: 2d audio array of shape (nsamples, 2).
    sr: sample rate.
    """
    t = np.arange(au.shape[0])/sr
    subplots(y, t, nrows=2, ncols=1, yaxis=0, title=title, subtitle=('left channel', 'right channel'), \
             xlabel='time (s)', ylabel='amplitude', grid=True, bgcolor=bgcolor, **kwargs)
