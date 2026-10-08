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

# change for each file
input_path='/Users/akshayr/Phys_871/Compton/Grad26/calibration/na22_withgain.mca'
output_path='/Users/akshayr/Phys_871/Compton/Grad26/analysis/na22_withgain.txt'

bkg_path='/Users/akshayr/Phys_871/Compton/Grad26/background_data_with_source_brick.mca'

def load_spectrum(path, n_channels=N_CHANNELS):
    header_data = []
    channel_counts = []
    channels = []
    is_data_section = False
    with open(path, "r", encoding="iso-8859-15") as file:
        i = 1
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
    ch, counts, live_time = load_spectrum(input_path)
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

    out = f"{output_path}_subtracted.txt"
    np.savetxt(out, np.column_stack([ch, subtracted]),
        header="subtracted counts", fmt=["%.5f", "%.0f"])
    print(f"Saved {out}")

if __name__ == "__main__":
    main()