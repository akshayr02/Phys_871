"""
Main file to perform the analysis

- Perform the peak fitting to find the peak energy for each angle (done)
- Calculate the energy over angle plots after background subtraction (done)
- Calculate yield ratio of the no target runs
- Plot 1/E' vs. 1-cos(theta), calculate find the line of best fit and get the slope (done)
- Calculate the differential cross section and plot against scattering angle
- Propagate all the uncertainties to the final results and error bars in final plots

Usage:
    python3 analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import scipy.odr as odr
import centroid_uncertainty
import calibration

FWHM_PER_SIGMA = 2.3548
LOG_Y = False
angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
hw = 100

# import calibration data from csv file
d = np.atleast_1d(np.genfromtxt("calib_results.csv", delimiter=",", names=True))
slope, slope_unc, intercept, intercept_unc, sys_err_keV = d["slope"], d["slope_unc"], d["intercept"], d["intercept_unc"], d["sys_err_keV"] 
coeffs = {}
coeffs["values"] = (slope[0], intercept[0])
coeffs["uncertainties"] = (slope_unc[0], intercept_unc[0])
coeffs["sys_err"] = sys_err_keV[0]

calib = np.poly1d(coeffs["values"])

energy_guesses=[631, 590, 640, 640, 640, 640, 640, 640, 640, 640, 640, 540, 500, 425, 414, 360, 300, 265]
# hws=[300, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 70, 150, 250, 150, 150, 100, 100]

PEAKS = {} # start off with guess of 2700 channel for angle=10

def energy_to_channel(energy):
    slope, intercept = coeffs["values"]
    print(slope, intercept)
    return (energy - intercept)/slope

for i in range(len(angles)):
    PEAKS[int(angles[i])] = (energy_to_channel(energy_guesses[i]), 150)
# print(PEAKS)

def load_spectrum(path):
    channel, counts = np.loadtxt(path, unpack=True)
    return channel, counts

def gauss(x, amp, mu, sigma, offset=0.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

def angular_error():
    d = 48.5-28.5
    R = 2.54
    angle_err = np.degrees(np.arctan(R/d))
    return angle_err

def quad_sum(*terms):
    """sqrt(sum of squares) of independent uncertainties."""
    return np.sqrt(sum(np.square(np.asarray(t, dtype=float)) for t in terms))

def channel_to_energy(ch, ch_err):
    slope, intercept = coeffs["values"]
    slope_unc, intercept_unc = coeffs["uncertainties"]

    peak_E = calib(ch)

    slope_unc_rel = slope_unc/slope
    channel_unc_rel = ch_err/ch
    peak_err_E = np.sqrt((slope*ch)**2 * ((slope_unc_rel)**2 + (channel_unc_rel)**2) + intercept_unc**2)
    
    return peak_E, peak_err_E

def fit_peak(ch, counts, guess, half_width, offset=False, sigma_y=None):
    """
    ch, counts : full spectrum
    guess      : rough peak channel (anywhere near the top is fine)
    half_width : window is guess +- half_width channels; use >= ~1.3 x FWHM
    offset     : also fit a constant background
    sigma_y    : optional per-channel errors for the FULL spectrum (see module docstring)
 
    Returns a dict with amp, mu, mu_err, fwhm, fwhm_err, popt, x, y, model.
    """
    sel = np.abs(ch - guess) <= half_width
    x, y = ch[sel], counts[sel]
 
    # start values taken from the data, not from the guess
    smooth = gaussian_filter1d(y, 5, mode="nearest")
    amp0, mu0 = smooth.max(), x[smooth.argmax()]
    sig0 = max(np.count_nonzero(smooth > amp0 / 2) / FWHM_PER_SIGMA, 3.0)
 
    p0 = [amp0, mu0, sig0] + ([0.0] if offset else [])
    lower = [0.0, x[0], 1.0] + ([-np.inf] if offset else [])
    upper = [10 * y.max(), x[-1], x[-1] - x[0]] + ([np.inf] if offset else [])
 
    popt, pcov = curve_fit(
        gauss, x, y, p0=p0, bounds=(lower, upper),
        sigma=None if sigma_y is None else np.asarray(sigma_y)[sel],
        absolute_sigma=sigma_y is not None, max_nfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return {
        "amp": popt[0], "mu": popt[1], "mu_err": perr[1],
        "fwhm": FWHM_PER_SIGMA * popt[2], "fwhm_err": FWHM_PER_SIGMA * perr[2],
        "popt": popt, "x": x, "y": y, "model": lambda xx: gauss(xx, *popt),
    }


def refine_centroid(channels, counts, guess, half_width):
    """Fit a Gaussian + linear background near `guess`.
    Returns (centroid, uncertainty); falls back to (guess, nan) on failure."""
    lo = max(int(guess - half_width), 0)
    hi = min(int(guess + half_width) + 1, len(counts))
    x = channels[lo:hi]
    y = counts[lo:hi]
    if len(x) < 6:
        return guess, np.nan

    p0 = [max(y) - min(y), guess, half_width / 3, 0.0, min(y)]
    sigma_y = np.sqrt(np.maximum(y, 1.0))  # Poisson errors
    try:
        popt, pcov = curve_fit(gauss_lin, x, y, p0=p0, absolute_sigma=False, maxfev=10000)
        perr = np.sqrt(pcov[1, 1])
        mu_err = perr[1]
        if not np.isfinite(mu_err) or abs(popt[1] - guess) > half_width:
            raise RuntimeError("unreliable fit")
        
        resid = (y - gauss_lin(x, *popt)) / sigma_y
        red_chi2 = np.sum(resid**2) / (len(x) - 5)
        mu_err_scaled = mu_err * np.sqrt(max(red_chi2, 1.0))
        fwhm = 2*np.sqrt(np.log(2)) * abs(popt[2])
        fwhm_err = 2*np.sqrt(np.log(2)) * perr[2]

        return popt[1], mu_err_scaled, fwhm_err, popt, x
    except (RuntimeError, ValueError):
        print(f"  Warning: Gaussian refit failed near channel {guess}; using given centroid.")
        return guess, np.nan

def main():
    angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
    peak_channels, channel_errs, peak_energies, energy_errs = [], [], [], []
    plot_configs = {}

    for (angle, (centroid_guess, hw)) in PEAKS.items():
        subtracted_file = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + str(angle) + ".txt"
        print(f"Loading " + subtracted_file)
        ch, counts = load_spectrum(subtracted_file)

        err = np.nan

        r = fit_peak(ch, counts, centroid_guess, hw)
            
        mean, err, fwhm, fwhm_err, popt = r["mu"], r["mu_err"], r["fwhm"], r["fwhm_err"], r["popt"]
        print("angle: ", str(angle))
        print("centroid: ", str(mean))
        print("fwhm: ", fwhm)
        
        combined_error = quad_sum(err, fwhm_err)
        
        peak_channels.append(mean)
        channel_errs.append(combined_error)

        peak_energy, energy_err = channel_to_energy(mean, combined_error)
        peak_energies.append(peak_energy)
        energy_errs.append(quad_sum(energy_err, coeffs["sys_err"])) # also add the systematic error from calib

        # plotting
        view = np.abs(ch - centroid_guess) <= 1.5 * hw
        x = ch[view]
        y = counts[view]
        
        fig, ax1 = plt.subplots()
        ax1.step(calib(x), y, where="mid", color="C7", alpha=0.7, linewidth=2, label='subtracted')
        # plot the gaussian fitting curve
        x_fit = np.linspace(r["x"][0], r["x"][-1], 500)
        y_fit = gauss(x_fit, *popt)
        ax1.plot(calib(x_fit), y_fit, color='C3', label='fit')
        ax1.set_ylabel("Counts")
        ax1.set_title("Subtracted counts with fitting for angle=" + str(angle), loc="left", fontsize=10)
        ax1.set_xlabel("Energy (keV)")
        
        ax2 = ax1.twiny()
        ax2.set_xlabel("Channels")
        ax2.step(x, y, where="mid", color="C7", alpha=0.7, linewidth=2, label='raw')
        ax2.plot(x_fit, y_fit, color='C3', label='fit')

        # store plot info in a dictionary to combine into one 3x3 plot for all angles and fits
        plot_configs[angle] = (calib(x), y, calib(x_fit), y_fit)
        ax2.grid(alpha=0.3)
        fig.savefig("plots/cs137_data_runs/cs137_angle_" + str(angle) + "_fitted.pdf", format="pdf", bbox_inches="tight")
        fig.clf()

    # remove the 21-29 angle measurements to get a better fit
    peak_channels = peak_channels[:2] + peak_channels[11:]
    channel_errs = channel_errs[:2] + channel_errs[11:]
    peak_energies = peak_energies[:2] + peak_energies[11:]
    energy_errs = energy_errs[:2] + energy_errs[11:]
    angles = angles[:2] + angles[11:]

    peak_channels = np.array(peak_channels, dtype=float)
    channel_errs = np.array(channel_errs, dtype=float)
    peak_energies = np.array(peak_energies, dtype=float)
    energy_errs = np.array(energy_errs, dtype=float)

    # print(peak_channels)
    # print(channel_errs)
    print(peak_energies.shape)
    print(energy_errs.shape)

    fig, axs = plt.subplots(3, 3)

    i = 0
    for (angle, (x, y, x_fit, y_fit)) in plot_configs.items():
        angle = str(angle)
        if angle in ["10", "20", "30"]:
            axs[0, i%3].step(x, y, color='C7', alpha=0.7, linewidth=2, label='background subtracted')
            axs[0, i%3].plot(x_fit, y_fit, color='C3', label='fit')
            axs[0, i%3].set_title('Angle = ' + angle + ' deg', fontsize=9)
        elif angle in ["40", "50", "60"]:
            axs[1, i%3].plot(x, y, color='C7', alpha=0.7, linewidth=2, label='background subtracted')
            axs[1, i%3].plot(x_fit, y_fit, color='C3', label='fit')
            axs[1, i%3].set_title('Angle = ' + angle + ' deg', fontsize=9)
        elif angle in ["70", "80", "90"]:
            axs[2, i%3].plot(x, y, color='C7', alpha=0.7, linewidth=2, label='background subtracted')
            axs[2, i%3].plot(x_fit, y_fit, color='C3', label='fit')
            axs[2, i%3].set_title('Angle = ' + angle + ' deg', fontsize=9)
        i += 1

    for ax in axs.flat:
        ax.set_xlabel('Energy (keV)', fontsize=7)
        ax.set_ylabel('Counts', fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7) 
        ax.grid(True, alpha=0.3)
        if LOG_Y:
            ax.set_yscale("log", nonpositive="clip")  # Applied to ax, not fig

    fig.tight_layout()
    fig.savefig("plots/cs137_data_runs/cs137_subtracted_counts_w_fitting.pdf", format="pdf", bbox_inches="tight")

    # plotting e over angle
    x = np.array(angles, dtype=int)
    angle_err = angular_error()/2

    fig, ax1 = plt.subplots()
    ax1.scatter(x, peak_energies, marker='o', label='data', zorder=2)
    ax1.errorbar(x, peak_energies, yerr=energy_errs, xerr=angle_err, label='error', fmt='none', color='C7', zorder=1)
    ax1.set_xlabel('Angle (deg)')
    ax1.set_ylabel('Energy (keV)')
    ax1.grid(alpha=0.3)
    ax1.set_title('Cs-137 Energy over Angle')
    ax1.legend()
    fig.tight_layout()
    fig.savefig("plots/energy_over_angle.pdf", format="pdf", bbox_inches="tight")
    fig.clf()

    # plotting inverse energy fit

    cos_x = np.array(1-np.cos(np.radians(x)), dtype=float)
    y=1/(peak_energies/1000) # convert to MeV so the y scale has less decimal places

    # first calculate the uncertainties for x and y error bars
    cos_x_errs = np.abs(np.sin(np.radians(x)))*np.radians(angle_err)
    y_errs = (energy_errs/peak_energies)*y

    # need to do a linear ODR fit since we have x and y errors

    # 1. Define a linear model: y = beta[0] + beta[1] * x
    linear_model = odr.Model(lambda beta, x: beta[0] + beta[1] * x)

    # 2. Package data and standard deviation errors into RealData
    data = odr.RealData(cos_x, y, sx=cos_x_errs, sy=y_errs)

    # 3. Set up ODR with initial parameter guesses [intercept, slope]
    odr_instance = odr.ODR(data, linear_model, beta0=[1.50, 1.96])

    # 4. Run the regression
    output = odr_instance.run()
    output.pprint()

    m, b = output.beta[1], output.beta[0]
    m_err, b_err = output.sd_beta[1], output.sd_beta[0]

    print(m, b)

    e_mass = 1/m
    e_mass_unc = ((m_err/m)*m)/2
    print(f"mc^2 = {e_mass:.3f} +/- {e_mass_unc:.3f} MeV")

    g_energy = 1/b
    g_energy_unc = ((b_err/b)*b)/2
    print(f"energy of incoming photon = {g_energy:.3f} +/- {g_energy_unc:.3f} MeV")

    fig, ax1 = plt.subplots()
    h_data = ax1.scatter(cos_x, y, marker='o', label='data', zorder=2)
    h_err = ax1.errorbar(cos_x, y, yerr=y_errs/2, xerr=cos_x_errs, label='error', fmt='none', color='C7', zorder=1)
    h_fit, = ax1.plot(cos_x, m * cos_x + b, color='red', label=f'Best fit: y = {m:.2f}x + {b:.2f}', zorder=3)
    ax1.set_xlabel('1-cos(theta)')
    ax1.set_ylabel("1/E'")
    ax1.grid(alpha=0.3)
    ax1.set_title('Cs-137 Compton energy shift')
    ax1.legend(loc="upper left", handles=[h_data, h_err, h_fit])
    fig.tight_layout()
    fig.savefig("plots/compton_energy_shift.pdf", format="pdf", bbox_inches="tight")
    
if __name__ == "__main__":
    main()



