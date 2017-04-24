#!/usr/bin/env python3
import sys
import pylga
import pymei
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse


def load_table(path):
    table = pylga.LGATable()
    table.load(open(path, 'rb'))
    data = [e for e in table]
    df = pd.DataFrame(data, columns=table.header)
    df['CDP'] = df.CDP.astype(int)
    df['HORIZON'] = df.HORIZON.astype(int)
    df.TIME /= 1000
    df.set_index('CDP', inplace=True)
    return df


def load_geom(path, az=None):
    if az is None:
        data = [(t.header.iline, t.cdp, t.mx, t.my, t.hx, t.hy) for t in pymei.load(path)]
        df = pd.DataFrame(data, columns=['iline', 'cdp', 'mx', 'my', 'hx', 'hy'])
        df.sort_values(['iline', 'cdp'], inplace=True)
        df = df.groupby(['iline', 'cdp']).mean()
        gb = df.groupby(level=0)
        delta = gb.last() - gb.first()
        az = np.mean(np.pi / 2 - np.arctan2(delta.my, delta.mx))
    ca = np.cos(az)
    sa = np.sin(az)
    data = [(t.cdp, t.mx * sa + t.my * ca) for t in pymei.load(path)]
    df = pd.DataFrame(data, columns=['CDP', 'm'])
    df.set_index('CDP', inplace=True)
    return df, az


def proj(df, az):
    df['a'] = (df.A0 * np.sin(az) + df.A1 * np.cos(az))
    del df['A0']
    del df['A1']
    return df


def background(path, geom, az, perc=0):
    ca = np.cos(az)
    sa = np.sin(az)
    data = sorted([(t.cdp, t.dt, t.data) for t in pymei.load(path)])
    dt = data[0][1]
    M = np.array([t for _, _, t in data]).T
    cdps = [c for c, _, _ in data]
    c0 = min(cdps)
    c1 = max(cdps)
    m0 = geom.loc[c0].m
    m1 = geom.loc[c1].m
    t0 = 0
    t1 = dt * M.shape[0]
    vmax = np.max(np.abs(M)) * (1 - perc/100)
    vmin = -vmax
    return M, [m0, m1, t0, t1], (vmin, vmax)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cmap', type=str, help='Color map for background image')
    parser.add_argument('-l', '--legend', action='store_true', help='Show color bar')
    parser.add_argument('-p', '--picks', type=str, help='Picks table')
    parser.add_argument('-t', '--title', type=str, help='Plot title')
    parser.add_argument('--dm', type=float, help='Size of slopes', default=150)
    parser.add_argument('--az', type=float, help='Data azimuth')
    parser.add_argument('--perc', type=int, help='Percentage gain', default=0)
    parser.add_argument('--clip', type=float, nargs='+', help='Clip the background image')
    parser.add_argument('--no-bg', dest='bg', action='store_false', help='Do not plot background')
    parser.add_argument('--coher', action='store_true', help='Set flags for coherence plotting')
    parser.add_argument('--stack', action='store_true', help='Set flags for stack plotting')
    parser.add_argument('--comp', action='store_true', help='Set flags for stack plotting')
    parser.add_argument('data', type=str, nargs='+', help='Reference geometry')
    parser.set_defaults(bg=True, legend=False)
    args = parser.parse_args()
    if args.coher:
        args.clip = [0, 1]
        args.cmap = 'inferno_r'
        args.legend = True
    if args.stack:
        args.cmap = 'gray'
        args.legend = True
    return args


class Plotter(object):
    def __init__(self, geom, az, dm, ax=None, share_ax=None):
        if not ax:
            fig = plt.figure()
            self.ax = fig.add_subplot(111, sharex=share_ax, sharey=share_ax)
            self.ax.invert_yaxis()
        else:
            self.ax = ax
        self.geom = geom
        self.az = az
        self.dm = dm

    def plot_picks(self, path):
        df = load_table(path)
        picks = proj(df, self.az)
        df = pd.merge(picks, self.geom, left_index=True, right_index=True)
        t1 = df.TIME - self.dm * df.a
        t2 = df.TIME + self.dm * df.a
        m1 = df.m - self.dm
        m2 = df.m + self.dm
        p1 = np.stack((m1, t1), axis=1)
        p2 = np.stack((m2, t2), axis=1)
        lines = LineCollection(np.stack((p1, p2), axis=1), color='red')
        self.ax.add_collection(lines)
        self.ax.scatter(df.m - self.dm, t1, color='red', marker='*', s=20)
        self.ax.scatter(df.m + self.dm, t2, color='red', marker='*', s=20)
        df.plot(kind='scatter', x='m', y='TIME', marker='o', s=30, ax=self.ax)

    def plot_background(self, path, clip=None, legend=False, **kwargs):
        bg, ext, (vmin, vmax) = background(path, self.geom, self.az, perc=0)
        if clip:
            vmin, vmax = clip
        mappable = self.ax.imshow(bg, extent=ext, aspect='auto',
                vmin=vmin, vmax=vmax, origin='lower', **kwargs)
        self.ax.set_xlim(ext[:2])
        self.ax.set_ylim(list(reversed(ext[2:])))
        if legend:
            self.ax.figure.colorbar(mappable)


if __name__ == '__main__':
    share_ax = None
    args = parse_args()
    for data in args.data:
        geom, az = load_geom(data, args.az)
        plotter = Plotter(geom, az, args.dm)
        #axes.append(plotter.ax)
        if args.bg:
            plotter.plot_background(data,
                    clip=args.clip,
                    legend=args.legend,
                    cmap=args.cmap)
        if args.picks:
            plotter.plot_picks(args.picks)
        if args.title:
            plotter.ax.set_title(args.title)
    plt.show()
