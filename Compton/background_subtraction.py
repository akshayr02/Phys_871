"""
Subtract the background from cosmics, residual light, and detector noise using the long background run.
Need to scale down the background to match the live time of the data runs

Usage:
    python background_subtraction.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================== CONFIG ====================================

N_CHANNELS = 4096
LOG_Y = False              # log scale for the spectra plots

SOURCES = [
    {
        "name": "Cs-137",
        "file_post": "/Users/akshayr/Phys_871/Compton/Grad26/analysis/cs137_withgain_subtracted.txt",
        "file_pre": "/Users/akshayr/Phys_871/Compton/Grad26/calibration/cs137_withgain.mca",
    },
    {
        "name": "Ba-133",
        "file_post": "/Users/akshayr/Phys_871/Compton/Grad26/analysis/ba133_withgain_subtracted.txt", #middle peak is optional
        "file_pre": "/Users/akshayr/Phys_871/Compton/Grad26/calibration/ba133_withgain.mca", #middle peak is optional
    },
    {
        "name": "Na-22",
        "file_post": "/Users/akshayr/Phys_871/Compton/Grad26/analysis/na22_withgain_subtracted.txt",
        "file_pre": "/Users/akshayr/Phys_871/Compton/Grad26/calibration/na22_withgain.mca",
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

bkg_path='/Users/akshayr/Phys_871/Compton/Grad26/calibration/bkg_withgain_850.mca'

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

        subtracted = []
        scale_factor = live_time/bkg_live_time
        print(scale_factor)
        print(len(ch))

        for channel in ch:
            new_val = counts[channel-1]-scale_factor*bkg_counts[channel-1]
            if new_val < 0:
                new_val = 0
            subtracted.append(new_val)

        spectra_pre[src["name"]] = (ch, counts)
        spectra_post[src["name"]] = (ch, subtracted)
        

    #plotting
    colors = {s["name"]: f"C{i}" for i, s in enumerate(SOURCES)}
    fig1, axes = plt.subplots(len(SOURCES), 1, figsize=(9, 3 * len(SOURCES)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, src in zip(axes, SOURCES):
        ch, counts_pre = spectra_pre[src["name"]]
        ch, counts_post = spectra_post[src["name"]]
        a.step(ch, counts_pre, where="mid", lw=0.8, color='C7', label='raw')
        a.step(ch, counts_post, where="mid", lw=0.8, color=colors[src["name"]], label='background subtracted')
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


    #save output
    # out = f"{SOURCES[index]["file_post"]}"
    # np.savetxt(out, np.column_stack([ch, subtracted]),
    #     header="subtracted counts", fmt=["%.5f", "%.0f"])
    # print(f"Saved {out}")

if __name__ == "__main__":
    main()