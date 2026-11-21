"""
Subtract the background from cosmics, residual light, and detector noise using the long background run.
Need to scale down the background to match the live time of the data runs
Used for calibration

Usage:
    python background_subtraction.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================== CONFIG ====================================

N_CHANNELS = 4096
LOG_Y = False              # log scale for the spectra plots

SOURCES = [
    {
        "name": "Cs-137",
        "file_post": "./Grad26/analysis/cs137_withgain_subtracted.txt",
        "file_pre": "./Grad26/calibration/cs137_withgain.mca",
    },
    {
        "name": "Ba-133",
        "file_post": "./Grad26/analysis/ba133_withgain_subtracted.txt", 
        "file_pre": "./Grad26/calibration/ba133_withgain.mca", 
    },
    {
        "name": "Na-22",
        "file_post": "./Grad26/analysis/na22_withgain_subtracted.txt",
        "file_pre": "./Grad26/calibration/na22_withgain.mca",
    },
]

# change for each file
source = "ba_133"

if source == "cs_137":
    index = 0
elif source == "ba_133":
    index = 1
if source == "na_22":
    index = 2

bkg_path='./Grad26/calibration/bkg_withgain_850.mca'

def load_spectrum(path, n_channels=N_CHANNELS):
    header_data = []
    channel_counts = []
    channels = []
    is_data_section = False
    with open(path, "r", encoding="iso-8859-15") as file:
        i = 0
        for line in file:
            cleaned_line = line.strip()
            
            # Identify where the raw channel data begins and ends
            if cleaned_line == "<<DATA>>":
                is_data_section = True
                continue
            elif cleaned_line == "<<END>>":
                is_data_section = False
                continue
                
            # Parse lines based on whether they are configuration metadata or channel counts
            if is_data_section:
                if cleaned_line:  # Ensure the line isn't empty
                    channel_counts.append(int(cleaned_line))
                    channels.append(i)
                    i+=1
            else:
                header_data.append(cleaned_line)
        for line in header_data:
            if "LIVE_TIME" in line:
                live_time = line[12:]
                print(live_time)
                break
    return channels, channel_counts, float(live_time)

def main():
    spectra_pre = {}
    spectra_post = {}
    for src in SOURCES:
        print(f"Loading {src['name']} from {src['file_pre']}")
        ch, counts, live_time = load_spectrum(src["file_pre"])
        bkg_ch, bkg_counts, bkg_live_time = load_spectrum(bkg_path)

        nbins = int(N_CHANNELS/16)
        
        counts, ch_edges = np.histogram(ch, bins=nbins, weights=counts)
        bkg_counts, bkg_ch_edges = np.histogram(bkg_ch, bins=nbins, weights=bkg_counts)
        
        # 2. Calculate bin centers and bin widths
        ch_binned, bkg_ch_binned = [], []
        
        for i in range(len(ch_edges) - 1):
            midpoint = (ch_edges[i] + ch_edges[i + 1]) / 2
            midpoint_2 = (bkg_ch_edges[i] + bkg_ch_edges[i + 1]) / 2
            ch_binned.append(midpoint)
            bkg_ch_binned.append(midpoint_2)
        widths = np.diff(ch_edges)
        print("bin width: ", widths[0])
        
        subtracted = []
        scale_factor = live_time/bkg_live_time
        print(scale_factor)
        print(len(ch_binned))

        for i in range(len(ch_binned)):
            new_val = counts[i]-scale_factor*bkg_counts[i]
            subtracted.append(new_val)

        spectra_pre[src["name"]] = (ch_binned, counts)
        spectra_post[src["name"]] = (ch_binned, subtracted)

        #save output
        j = np.arange(0, len(ch_binned), 1)
        out = f"{src["file_post"]}"
        np.savetxt(out, np.column_stack([j, ch_binned, subtracted]),
            header="subtracted counts", fmt=["%i", "%.5f", "%.0f"])
        print(f"Saved {out}")
        

    #plotting
    colors = {s["name"]: f"C{i}" for i, s in enumerate(SOURCES)}
    fig1, axes = plt.subplots(len(SOURCES), 1, figsize=(9, 3 * len(SOURCES)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, src in zip(axes, SOURCES):
        ch_binned, counts_pre = spectra_pre[src["name"]]
        ch_binned, counts_post = spectra_post[src["name"]]
        a.step(ch_binned, counts_pre, where="mid", lw=0.8, color='C7', label='raw')
        a.step(ch_binned, counts_post, where="mid", lw=0.8, color=colors[src["name"]], label='background subtracted')
        if LOG_Y:
            a.set_yscale("log")
        a.set_ylabel("Counts")
        a.set_title(src["name"], loc="left", fontsize=10)
        a.grid(alpha=0.3)
        a.legend()
    axes[-1].set_xlabel("Channel")
    fig1.tight_layout()
    plt.savefig("plots/background_subtraction.pdf", format="pdf", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()