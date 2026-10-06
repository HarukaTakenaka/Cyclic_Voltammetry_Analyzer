"""
cv_peakratio_biologic.py
========================
All helper functions and classes for CV peak-ratio analysis (BioLogic .mpr files).
Import this from the notebook — do not edit the notebook itself for logic changes.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log
from scipy.signal import filtfilt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from galvani import BioLogic as BL


# ── Utility ────────────────────────────────────────────────────────────────

def enum(group):
    """Enumerate a list or dict, always returning [[key, value], ...]."""
    if isinstance(group, dict):
        return [[k, v] for k, v in group.items()]
    return [[i, v] for i, v in enumerate(group)]


def prox(target, aList):
    """Return the closest value to target in aList."""
    return min(aList, key=lambda x: abs(x - target))


def getCscale(cUnits):
    """
    Return (scale_factor, unit_label) for current display.
    BioLogic stores current in the column units directly (mA or uA),
    so scale_factor here is just for display consistency.
      'mA'  → (1.0,    'mA')
      'uA'  → (1.0,    'µA')   # already in uA from the MPR column
    """
    if cUnits == 'uA':
        return 1.0, r'$\mu$A'
    return 1.0, 'mA'


def prepName(filename, replacers):
    """Strip .mpr extension and apply replacer dict to get a clean name."""
    name = filename.replace('.mpr', '')
    for key, val in replacers.items():
        name = name.replace(key, val)
    return name


def findScanRate(name):
    """Extract scan rate (first integer) from a filename string."""
    hits = re.findall(r'\d+', name)
    if not hits:
        raise ValueError(f'No number found in filename: {name}')
    if len(hits) != 1:
        print(f'WARNING: multiple numbers in "{name}", using first: {hits[0]}')
    return int(hits[0])


# ── Data class ─────────────────────────────────────────────────────────────

class aRun:
    """
    Represents one .mpr file (one scan rate).
    Data is split into cycles, peaks are found per cycle.
    """

    def __init__(self, name, scanRate, Cs, Vs, cycleNums, Ts,
                 FpeakRange, RpeakRange):
        self.name        = name
        self.scanRate    = scanRate
        self.Cs          = Cs
        self.Vs          = Vs
        self.cycleNums   = cycleNums
        self.Ts          = Ts
        self.FpeakRange  = FpeakRange
        self.RpeakRange  = RpeakRange
        self.cycleCs     = []
        self.cycleVs     = []
        self.cycleTs     = []
        self.cyclePeaks  = []
        self.dataToCycles()
        self.findPeaks()

    def __repr__(self):
        return '{run ' + self.name + '}'

    def __str__(self):
        return '{run ' + self.name + '}'

    def dataToCycles(self):
        """Split flat Cs/Vs/Ts arrays into per-cycle sublists."""
        cycleValues  = [int(i) for i in sorted(set(self.cycleNums))]
        cycleStarts  = [self.cycleNums.index(v) for v in cycleValues] + [-1]
        last = 0
        for start in cycleStarts[1:]:
            self.cycleCs.append(self.Cs[last:start])
            self.cycleVs.append(self.Vs[last:start])
            self.cycleTs.append(self.Ts[last:start])
            last = start

    def findPeaks(self):
        """
        Find forward and reverse peaks for each cycle.
        Uses self.FpeakRange / self.RpeakRange if provided; otherwise searches the full segment.
        """
        Fyes = bool(self.FpeakRange)
        Ryes = bool(self.RpeakRange)

        for cyclei, _ in enum(self.cycleCs):
            segLen = len(self.cycleCs[cyclei]) // 2
            FVs = self.cycleVs[cyclei][:segLen]
            RVs = self.cycleVs[cyclei][segLen:]
            FCs = self.cycleCs[cyclei][:segLen]
            RCs = self.cycleCs[cyclei][segLen:]

            # Make both peak segments positive so argmax works uniformly
            if self.cycleCs[cyclei][segLen] < 0:
                FCs = [-c for c in FCs]
            else:
                RCs = [-c for c in RCs]

            # Determine search ranges
            if Fyes:
                FRange = [prox(self.FpeakRange[0], FVs), prox(self.FpeakRange[1], FVs)]
            else:
                FRange = [FVs[0], FVs[-1]]

            if Ryes:
                RRange = [prox(self.RpeakRange[0], RVs), prox(self.RpeakRange[1], RVs)]
            else:
                RRange = [RVs[0], RVs[-1]]

            FiRange = [FVs.index(FRange[0]), FVs.index(FRange[1])]
            RiRange = sorted([RVs.index(RRange[0]), RVs.index(RRange[1])])

            Fpeak  = max(FCs[FiRange[0]:FiRange[1]])
            Rpeak  = max(RCs[RiRange[0]:RiRange[1]])
            FpeakV = FVs[FCs.index(Fpeak)]
            RpeakV = RVs[RCs.index(Rpeak)]

            self.cyclePeaks.append({FpeakV: Fpeak, RpeakV: Rpeak})


# ── File loading ───────────────────────────────────────────────────────────

def findSubdirNames(directory):
    """Return names of all immediate subdirectories inside directory."""
    return [e for e in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, e))]


def findMPRs(fileFolderPath, fileFolderName):
    """Find all .mpr files inside fileFolderPath. Returns list of filenames or None."""
    print('*' * 80 + f'\n\nsearching folder {fileFolderName} at {fileFolderPath}')
    MPRfileNames = []
    for root, dirs, files in os.walk(fileFolderPath):
        found = [f for f in files if f.endswith('.mpr')]
        if found:
            print('found MPR files:\n', '\t'.join(found))
            MPRfileNames = found
    if not MPRfileNames:
        print(f'\nWARNING: folder {fileFolderName} contained no MPR files, skipping.\n')
        return None
    return MPRfileNames


def extractMPRs(MPRfileNames, fileFolderPath, replacers, cUnits,
                noiseFactor, FpeakRange, RpeakRange):
    """
    Load .mpr files and return a sorted list of aRun objects.

    Parameters
    ----------
    MPRfileNames  : list of .mpr filenames
    fileFolderPath: folder containing those files
    replacers     : dict of string replacements for filename cleanup
    cUnits        : 'mA' or 'uA' — selects the current column from the MPR
    noiseFactor   : integer; higher = more smoothing (filtfilt window size)
    FpeakRange    : [lo, hi] voltage window for forward peak; [] = whole segment
    RpeakRange    : [lo, hi] voltage window for reverse peak; [] = whole segment
    """
    fA = 1
    fB = [1.0 / noiseFactor] * noiseFactor

    runs = []
    for fname in MPRfileNames:
        fpath = os.path.join(fileFolderPath, fname)
        mpr   = BL.MPRfile(fpath)
        df    = pd.DataFrame(mpr.data)

        name     = prepName(fname, replacers)
        scanRate = findScanRate(name)

        if scanRate == 0:
            print(f'\nWARNING: {fname} has scan rate 0 — setting to 1. Fix filename and rerun.\n')
            scanRate = 1

        Vs = df['Ewe/V']
        # Handle both averaging (<I>/mA) and instantaneous (I/mA) column names
        col = f'<I>/{cUnits}' if f'<I>/{cUnits}' in df.columns else f'I/{cUnits}'
        Cs  = df[col]

        Cs = list(filtfilt(fB, fA, Cs))
        Vs = list(filtfilt(fB, fA, Vs))

        cycleNums = list(df['cycle number'])
        Ts        = list(df['time/s'])

        run = aRun(name, scanRate, Cs, Vs, cycleNums, Ts, FpeakRange, RpeakRange)
        runs.append(run)
        print(f'  {fname} → {scanRate} mV/s, cycles={len(run.cyclePeaks)}')

    return sorted(runs, key=lambda r: r.scanRate)


def makeOutDir(fileFolderPath):
    """Create and return an 'output' subdirectory inside fileFolderPath."""
    d = os.path.join(fileFolderPath, 'output')
    os.makedirs(d, exist_ok=True)
    return d


# ── Plotting ───────────────────────────────────────────────────────────────

def plotCVs(runs, outDir, goTexas, refElectrode, cUnits,
            only_cycle=None, save_plot=True):
    """
    Plot all scan rates for each cycle (overview, no peak markers).

    Parameters
    ----------
    only_cycle : int or None. If set (1-indexed), plots only that cycle.
    """
    _, cunit_label = getCscale(cUnits)
    E       = r'$\mathit{E}$'
    i_label = r'$\mathit{i}$'

    cycleNum = max(len(r.cyclePeaks) for r in runs)
    cycles   = [only_cycle - 1] if only_cycle else range(cycleNum)

    allCs = [c for r in runs for c in r.Cs]
    if goTexas:
        allCs = [-c for c in allCs]
    Cmax, Cmin = max(allCs), min(allCs)

    for cyclei in cycles:
        plt.figure(figsize=(10, 7), tight_layout=True)

        for runi, run in enum(runs):
            try:
                Vs = run.cycleVs[cyclei]
                Cs = run.cycleCs[cyclei]
            except IndexError:
                print(f'Run {run.name} has no data for cycle {cyclei + 1}')
                continue
            if goTexas:
                Cs = [-c for c in Cs]
            color = cm.get_cmap('rainbow')(runi / len(runs))
            plt.plot(Vs, Cs, label=run.name, color=color)

        plt.ylim(top=Cmax * 1.01, bottom=Cmin * 1.01)
        plt.xlabel(f'Potential, {E} (V vs. {refElectrode})')
        plt.ylabel(f'Current, {i_label} ({cunit_label})')
        plt.title(f'Cycle {cyclei + 1}', fontweight='extra bold')
        plt.legend(fontsize=8, framealpha=1, edgecolor='black', loc='upper left')
        plt.minorticks_on()
        if goTexas:
            plt.gca().invert_xaxis()
        if save_plot:
            plt.savefig(os.path.join(outDir, f'cycle_{cyclei+1}.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Displayed figure for cycle {cyclei + 1}')


def plotPeakCheck(runs, outDir, goTexas, refElectrode, cUnits,
                  FpeakRange, RpeakRange, drawRanges=True,
                  only_cycle=None, save_plot=True):
    """
    Plot CVs with detected peak markers and optional range shading.

    Peak markers:
      ▲  forward (reduction) peak  — dark blue
      ▼  reverse (oxidation) peak  — dark red

    Range shading (when drawRanges=True):
      Blue band = FpeakRange search window
      Red  band = RpeakRange search window
    If either range is [] the shading for that range is skipped.
    """
    _, cunit_label = getCscale(cUnits)
    E       = r'$\mathit{E}$'
    i_label = r'$\mathit{i}$'

    cycleNum = max(len(r.cyclePeaks) for r in runs)
    cycles   = [only_cycle - 1] if only_cycle else range(cycleNum)

    allCs = [c for r in runs for c in r.Cs]
    if goTexas:
        allCs = [-c for c in allCs]
    Cmax, Cmin = max(allCs), min(allCs)

    for cyclei in cycles:
        fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)

        # ── Range shading ──
        if drawRanges:
            if FpeakRange and len(FpeakRange) == 2:
                ax.axvspan(min(FpeakRange), max(FpeakRange),
                           color='#3a3f99', alpha=0.10, label='Forward peak range')
                ax.axvline(min(FpeakRange), color='#3a3f99', lw=0.8, ls='--', alpha=0.5)
                ax.axvline(max(FpeakRange), color='#3a3f99', lw=0.8, ls='--', alpha=0.5)
            if RpeakRange and len(RpeakRange) == 2:
                ax.axvspan(min(RpeakRange), max(RpeakRange),
                           color='#c0392b', alpha=0.10, label='Reverse peak range')
                ax.axvline(min(RpeakRange), color='#c0392b', lw=0.8, ls='--', alpha=0.5)
                ax.axvline(max(RpeakRange), color='#c0392b', lw=0.8, ls='--', alpha=0.5)

        # ── CV lines + peak markers ──
        for runi, run in enum(runs):
            try:
                Vs = run.cycleVs[cyclei]
                Cs = run.cycleCs[cyclei]
            except IndexError:
                print(f'Run {run.name} has no data for cycle {cyclei + 1}')
                continue

            if goTexas:
                Cs = [-c for c in Cs]
            color = cm.get_cmap('rainbow')(runi / len(runs))
            ax.plot(Vs, Cs, color=color, alpha=0.7)

            peaks  = run.cyclePeaks[cyclei]
            peakVs = list(peaks.keys())
            peakCs = [peaks[v] for v in peakVs]
            FpeakV, FpeakC = peakVs[0], peakCs[0]
            RpeakV, RpeakC = peakVs[1], peakCs[1]

            FpeakC_plot = FpeakC  if goTexas else -FpeakC
            RpeakC_plot = -RpeakC if goTexas else  RpeakC

            ax.plot(FpeakV, FpeakC_plot,
                    marker='^', color='#1a237e', ms=10, zorder=5, lw=0,
                    label='Forward peak' if runi == 0 else '')
            ax.plot(RpeakV, RpeakC_plot,
                    marker='v', color='#b71c1c', ms=10, zorder=5, lw=0,
                    label='Reverse peak' if runi == 0 else '')

        ax.set_ylim(top=Cmax * 1.01, bottom=Cmin * 1.01)
        ax.set_xlabel(f'Potential, {E} (V vs. {refElectrode})')
        ax.set_ylabel(f'Current, {i_label} ({cunit_label})')
        ax.set_title(f'Cycle {cyclei + 1} — Peak Check', fontweight='extra bold')
        ax.legend(fontsize=9, loc='upper left')
        if goTexas:
            ax.invert_xaxis()
        if save_plot:
            fig.savefig(os.path.join(outDir, f'cycle_{cyclei+1}_peakcheck.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()
        print(f'Displayed figure for cycle {cyclei + 1}')


# ── Excel export ───────────────────────────────────────────────────────────

def exportPeakExcel(runs, outDir, nequiv):
    """
    Compute peak ratios and kinetic quantities for each cycle, save to Excel.
    Returns the DataFrame for the last cycle.
    """
    cycleNum = max(len(r.cyclePeaks) for r in runs)
    df_last  = None

    for cyclei in range(cycleNum):
        header = ['name', 't', '1/((n-1)[Ni]0)*ln(m)', 'scan rate',
                  'Epa', 'Epc', 'ipa', 'ipc',
                  'ia/ic', 'm=1/nx(n-1)/n/(ia/ic)', '1/.001*(ia/ic)-1/.001', 'Vswitch']
        rows   = []
        eq4s, eq5s = [], []

        for run in runs:
            try:
                peaks = run.cyclePeaks[cyclei]
            except IndexError:
                print(f'Run {run.name} has no data for cycle {cyclei + 1}')
                continue

            peakVs  = list(peaks.keys())
            peakCs  = [peaks[v] for v in peakVs]
            FpeakC, FpeakV = peakCs[0], peakVs[0]
            RpeakC, RpeakV = peakCs[1], peakVs[1]

            Vswitch = _findVswitch(run.cycleVs[0])
            eq1 = RpeakC / FpeakC
            eq2 = (1 / nequiv) + (((nequiv - 1) / nequiv) / eq1)
            eq3 = 0
            eq4 = ((RpeakV - Vswitch) + (FpeakV - Vswitch)) / (run.scanRate / 1000)
            if eq2 < 0:
                print(f'WARNING: negative eq2 for {run.name}, cycle {cyclei + 1}')
                eq5 = -10000
            else:
                eq5 = (1 / ((nequiv - 1) * 0.001)) * log(eq2)

            rows.append([run.name, eq4, eq5, run.scanRate,
                         RpeakV, FpeakV, RpeakC, FpeakC,
                         eq1, eq2, eq3, Vswitch])
            if str(run.scanRate) == run.name:
                eq4s.append(eq4)
                eq5s.append(eq5)

        df_out = pd.DataFrame(rows, columns=header)
        fname  = os.path.join(outDir, f'cycle {cyclei + 1}.xlsx')
        written = False
        while not written:
            try:
                df_out.to_excel(fname, index=False)
                print(f'Saved: cycle {cyclei + 1}.xlsx')
                written = True
            except PermissionError:
                input(f'Close the open Excel file and press ENTER:\n  {fname}')
        df_last = df_out

    return df_last


def _findVswitch(Vs):
    """Find the switching potential (most extreme V) near the midpoint of a cycle."""
    halfPoint   = len(Vs) // 2
    switchRange = Vs[halfPoint - 10: halfPoint + 10]
    Max, Min    = max(Vs), min(Vs)
    if Max in switchRange:
        return Max
    return Min


# ── Regression plot ────────────────────────────────────────────────────────

def regressionPlot(df, outDir, nequiv, ligname='', substrate_name='',
                   exclude=[], title_suffix='', savename='plot', save_plot=True):
    """
    Linear regression of kinetic data with optional scan-rate exclusion.
    Excluded points are NOT shown — plot auto-scales to remaining data only.

    Returns
    -------
    (slope, intercept, r2)
    """
    x_axis    = df['t'].values
    y_axis    = df['1/((n-1)[Ni]0)*ln(m)'].values
    scan_rate = df['scan rate'].values

    mask = (
        ~np.isnan(x_axis) & ~np.isnan(y_axis) &
        np.isfinite(x_axis) & np.isfinite(y_axis) &
        ~np.isin(scan_rate, exclude)
    )
    x_clean         = x_axis[mask].reshape(-1, 1)
    y_clean         = y_axis[mask]
    scan_rate_clean = scan_rate[mask]

    model     = LinearRegression().fit(x_clean, y_clean)
    slope     = model.coef_[0]
    intercept = model.intercept_
    y_pred    = model.predict(x_clean)
    r2        = r2_score(y_clean, y_pred)

    plt.figure(figsize=(12, 8))
    plt.scatter(x_clean, y_clean, color='#0d48a6', marker='x', zorder=5, s=80)
    plt.plot(x_clean, y_pred, color='gray', linestyle='-', alpha=0.8)

    for k in range(len(x_clean)):
        plt.annotate(f'{scan_rate_clean[k]:.0f}',
                     (x_clean[k], y_clean[k]),
                     textcoords='offset points', xytext=(0, -14),
                     ha='center', fontsize=12)

    plt.xlabel('t', fontsize=14)
    plt.ylabel('1/((n-1)[Ni]0)*ln(m)', fontsize=14)
    plt.title(f'{ligname}  {substrate_name}  {title_suffix}', fontsize=16)
    plt.legend([f'y = {slope:.2f}x + {intercept:.2f}\nabs rate: {slope:.2f}\nR² = {r2:.3f}'],
               fontsize=14)
    plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(outDir, f'{savename}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f'{nequiv} equiv  |  Slope: {slope:.4f}  |  Intercept: {intercept:.4f}  |  R²: {r2:.4f}')
    return slope, intercept, r2
