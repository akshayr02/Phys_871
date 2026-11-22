"""
Subtract the background from cosmics, residual light, and detector noise using no target runs for each angle measurement
Also makes plots of the data before and after subtraction for each angle
Usage:
    python bkg_subtraction_angle.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# =============================== CONFIG ====================================

N_CHANNELS = 4096
LOG_Y = False              # log scale for the spectra plots

# angles = ["0", "10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
angles = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]

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
    spectra_pre, spectra_post, spectra_nt = {}, {}, {}
    for i in range(len(angles)):
        target_file = "Grad26/cs137/cs137_source_angle_" + angles[i] + "_target.mca"
        no_target_file = "Grad26/cs137/cs137_source_angle_" + angles[i] + "_no_target.mca"
        print(f"Loading " + target_file)
        print(f"Loading " + no_target_file)
        ch, counts, live_time = load_spectrum(target_file)
        ch_nt, counts_nt, live_time2 = load_spectrum(no_target_file)

        # if os.path.isfile(no_target_file): # use the long background run if we don't have no target runs
        #     bkg_ch, bkg_counts, bkg_live_time = load_spectrum(no_target_file)
        # else:
        #     bkg_ch, bkg_counts, bkg_live_time = load_spectrum(bkg_path)

        bkg_ch, bkg_counts, bkg_live_time = load_spectrum(bkg_path)

        nbins = int(N_CHANNELS/50)
                
        counts, ch_edges = np.histogram(ch, bins=nbins, weights=counts)
        bkg_counts, bkg_ch_edges = np.histogram(bkg_ch, bins=nbins, weights=bkg_counts)
        counts_nt, ch_nt_edges = np.histogram(ch_nt, bins=nbins, weights=counts_nt)
                
        # 2. Calculate bin centers and bin widths
        ch_binned, bkg_ch_binned, ch_nt_binned = [], [], []

        print(len(ch_nt_edges), len(ch_edges))        
        for j in range(len(ch_edges) - 1):
            midpoint = (ch_edges[j] + ch_edges[j + 1]) / 2
            midpoint_2 = (bkg_ch_edges[j] + bkg_ch_edges[j + 1]) / 2
            midpoint_3 = (ch_nt_edges[j] + ch_nt_edges[j + 1]) / 2
            ch_binned.append(midpoint)
            bkg_ch_binned.append(midpoint_2)
            ch_nt_binned.append(midpoint_3)
        widths = np.diff(ch_edges)
        print("bin width: ", widths[0])
        
        scale_factor = live_time/bkg_live_time
        scale_factor2 = live_time2/bkg_live_time
        print(scale_factor2)
        # print(len(ch_binned))

        subtracted = []
        for j in range(len(ch_binned)):
            new_val = counts[j]-scale_factor*bkg_counts[j]
            subtracted.append(new_val)

        subtracted_nt = []
        for j in range(len(ch_nt_binned)):
            new_val = counts[j]-scale_factor2*bkg_counts[j]
            subtracted_nt.append(new_val)

        spectra_pre[angles[i]] = (ch_binned, counts)
        spectra_post[angles[i]] = (ch_binned, subtracted)
        spectra_nt[angles[i]] = (ch_nt_binned, subtracted_nt)

        # save output
        out = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + angles[i] + ".txt"
        np.savetxt(out, np.column_stack([ch_binned, subtracted]), header="subtracted counts", fmt=["%.5f", "%.0f"])
        print(f"Saved {out}")

        out = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + angles[i] + "_no_target.txt"
        np.savetxt(out, np.column_stack([ch_nt_binned, subtracted_nt]), header="subtracted nt counts", fmt=["%.5f", "%.0f"])
        print(f"Saved {out}")
        
        # plotting
        if i == 1: # remove the spike for 20 deg
            spectra_pre[angles[i]] = (ch_binned[10:], counts[10:])
            spectra_post[angles[i]] = (ch_binned[10:], subtracted[10:])
            spectra_nt[angles[i]] = (ch_nt_binned[10:], subtracted_nt[10:])
        
        (x, y) = spectra_post[angles[i]]
        (x2, y2) = spectra_pre[angles[i]]
        (x3, y3) = spectra_pre[angles[i]]
        plt.plot(x2, y2, color="C7", alpha=0.7, linewidth=2, label='raw')
        # plt.step(bkg_ch_binned, bkg_counts*scale_factor, color='C2', alpha=0.7, linewidth=2, label='background')
        plt.plot(x, y, color='C0', alpha=0.7, linewidth=2, label='subtracted')
        # plt.axvline(2283, color="gray")
        # plt.axvline(2903, color="red")

        plt.title('Cs-137 angle measurements (' + angles[i] + ' deg)')
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.grid(alpha=0.3)
        plt.legend()
        if LOG_Y:
            plt.yscale("log", nonpositive="clip")
        plt.savefig("plots/cs137_data_runs/cs137_angle_" + angles[i] + ".pdf", format="pdf", bbox_inches="tight")
        plt.clf()

        # plot target/no target plots
        plt.plot(x3, y3, color="C2", alpha=0.7, linewidth=2, label='no target subtracted')
        plt.plot(x, y, color='C0', alpha=0.7, linewidth=2, label='target subtracted')
        plt.title('Cs-137 angle measurements (' + angles[i] + ' deg) no target comparison')
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.grid(alpha=0.3)
        plt.legend()
        if LOG_Y:
            plt.yscale("log", nonpositive="clip")
        plt.savefig("plots/cs137_data_runs/cs137_angle_" + angles[i] + "_no_target.pdf", format="pdf", bbox_inches="tight")
        plt.clf()



        # plt.show()

        

    # make a combined subplot of all the angles
    fig, axs = plt.subplots(3, 3)

    i=0
    for angle in spectra_post:
        (x, y) = spectra_post[angle]
        (x2, y2) = spectra_pre[angle]
        if angle=="10" or angle=="20" or angle=="30":
            axs[0, i%3].plot(x, y, color='C0', alpha=0.7, linewidth=2, label='background subtracted')
            axs[0, i%3].set_title('Angle = ' + angle + ' deg')
        if angle=="40" or angle=="50" or angle=="60":
            axs[1, i%3].plot(x, y, color='C0', alpha=0.7, linewidth=2, label='background subtracted')
            axs[1, i%3].set_title('Angle = ' + angle + ' deg')
        if angle=="70" or angle=="80" or angle=="90":
            axs[2, i%3].plot(x, y, color='C0', alpha=0.7, linewidth=2, label='background subtracted')
            axs[2, i%3].set_title('Angle = ' + angle + ' deg')
        i+=1

    for ax in axs.flat:
        ax.set(xlabel='Channel', ylabel='Counts')
        ax.label_outer()
        ax.grid(True, alpha=0.3)
    if LOG_Y:
        fig.set_yscale("log", nonpositive="clip")
    fig.tight_layout()
    
    fig.savefig("plots/cs137_data_runs/cs137_subtracted_counts.pdf", format="pdf", bbox_inches="tight")
    fig.clf()


if __name__ == "__main__":
    main()