"""
Main file to perform the analysis

- Perform the peak fitting to find the peak energy for each angle (done)
- Plot the Gaussian fitting to the data for each angle (done)
- Calculate and plot the energy over angle after background subtraction (done)
- Calculate yield ratio of the no target runs (skip)
- Plot 1/E' vs. 1-cos(theta), calculate find the line of best fit and get the slope and intercept (done)
- Calculate the yield under peak and uncertainty (done)
- Calculate the differential cross section and plot against scattering angle (done)
- Propagate all the uncertainties to the final results and error bars in final plots (done)

Usage:
    python3 analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from odrpack import odr_fit

FWHM_PER_SIGMA = 2.3548
LOG_Y = False
# angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
angles = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
hw = 100
global r0 
r0 = 2.82*10**(-13)

# import calibration data from csv file
d = np.atleast_1d(np.genfromtxt("calib_results.csv", delimiter=",", names=True))
slope, slope_unc, intercept, intercept_unc, sys_err_keV = d["slope"], d["slope_unc"], d["intercept"], d["intercept_unc"], d["sys_err_keV"] 
coeffs = {}
coeffs["values"] = (slope[0], intercept[0])
coeffs["uncertainties"] = (slope_unc[0], intercept_unc[0])
coeffs["sys_err"] = sys_err_keV[0]

calib = np.poly1d(coeffs["values"])

# used for reference, now using energy guesses from compton angle and half-width from scaling down the true peak based on sqrt(E)
energy_guesses=[650, 590, 640, 640, 640, 640, 640, 640, 640, 640, 640, 540, 500, 425, 414, 360, 300, 283]
hws=[150, 150, 100, 100, 100, 100, 100, 100, 100, 100, 100, 150, 150, 150, 150, 150, 150, 150]

PEAKS = {} # start off with guess of 2700 channel for angle=10


"""
Bounded Gaussian fit for weak peaks in background-subtracted spectra.

The old free fit lets sigma shrink to a single channel (it then fits a noise spike) or
grow until the Gaussian is a flat baseline. Here the peak position and width are tied to
physics: Compton kinematics for where the peak is, and the detector resolution for how
wide it is. The yield comes out in COUNTS (not counts*keV) with an error from the full
covariance matrix. Keep negative values in the subtracted spectrum (no np.clip).
"""

E0_KEV = 661.657
MC2_KEV = 510.999

FWHM_662 = 44.8       # keV, FWHM of the Cs-137 peak in your calibration spectrum

def compton_energy(angle_deg, E0=E0_KEV):
    return E0 / (1.0 + (E0 / MC2_KEV) * (1.0 - np.cos(np.radians(angle_deg))))

def expected_peak(angle_deg):
    """(expected energy, expected FWHM) in keV; NaI-type resolution scales as sqrt(E)."""
    E = compton_energy(angle_deg)
    return E, FWHM_662 * np.sqrt(E / E0_KEV)

def gauss(x, amp, mu, sigma, offset=0.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

def fit_peak_bounded(x, counts, guess, fwhm_exp, bin_width, offset=True,
                     sigma_y=None, mu_tol=20.0, width_range=(0.6, 1.6), hw_factor=2.0):
    """
    x          : energy axis of the full spectrum (keV)
    guess      : expected peak energy (keV), e.g. from expected_peak(angle)
    fwhm_exp   : expected FWHM at that energy (keV)
    bin_width  : keV per channel; converts the area into counts
    mu_tol     : the centroid may move +- mu_tol keV from the guess
    width_range: sigma may range over this multiple of the expected sigma
    hw_factor  : fit window = guess +- hw_factor * fwhm_exp

    Returns a dict with yield (counts), yield_err, mu, mu_err, fwhm, fwhm_err, popt, x, y, model.
    A yield smaller than ~3 x yield_err means the peak is not detected.
    """
    x = np.asarray(x, float)
    counts = np.asarray(counts, float)
    sel = np.abs(x - guess) <= hw_factor * fwhm_exp
    # direct = (y[sel] - offset).sum() 
    # print(sel, direct)
    xs, ys = x[sel], counts[sel]
    sig_exp = fwhm_exp / FWHM_PER_SIGMA

    core = ys[np.abs(xs - guess) < sig_exp]
    p0 = [max(core.mean(), 1e-3), guess, sig_exp] + ([0.0] if offset else [])
    lower = [0.0, guess - mu_tol, width_range[0] * sig_exp] + ([-np.inf] if offset else [])
    upper = [np.inf, guess + mu_tol, width_range[1] * sig_exp] + ([np.inf] if offset else [])

    popt, pcov = curve_fit(
        gauss, xs, ys, p0=p0, bounds=(lower, upper),
        sigma=None if sigma_y is None else np.asarray(sigma_y, float)[sel],
        absolute_sigma=sigma_y is not None, max_nfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    # calculate yield under Gaussian peak in energy axis
    scale = np.sqrt(2 * np.pi) / bin_width                 # (counts/bin x keV) -> counts
    yield_ = popt[0] * popt[2] * scale                     # A * sigma * sqrt(2 pi) / bin_width
    J = np.array([popt[2], 0.0, popt[0]]) * scale          # d(yield)/d(amp, mu, sigma)
    yield_err = np.sqrt(J @ pcov[:3, :3] @ J)              # err = sqrt(variance) = sqrt(J^T C J) 

    return {
        "amp": popt[0], "mu": popt[1], "mu_err": perr[1],
        "fwhm": FWHM_PER_SIGMA * popt[2], "fwhm_err": FWHM_PER_SIGMA * perr[2],
        "popt": popt, "x": x, "y": ys, "model": lambda xx: gauss(xx, *popt),
        "yield": (yield_, yield_err)
    }


def energy_to_channel(energy):
    slope, intercept = coeffs["values"]
    return (energy - intercept)/slope

for i in range(len(angles)):
    PEAKS[int(angles[i])] = (energy_to_channel(energy_guesses[i]), hws[i])
# print(PEAKS)

def load_spectrum(path):
    channel, counts = np.loadtxt(path, unpack=True)
    return channel, counts

def angular_error():
    d = 48.5-28.5
    R = 2.54
    angle_err = np.degrees(np.arctan(R/d))
    return angle_err

def quad_sum(*terms):
    """sqrt(sum of squares) of independent uncertainties."""
    return np.sqrt(sum(np.square(np.asarray(t, dtype=float)) for t in terms))

# def channel_to_energy(ch, ch_err):
#     slope, intercept = coeffs["values"]
#     slope_unc, intercept_unc = coeffs["uncertainties"]

#     peak_E = calib(ch)

#     slope_unc_rel = slope_unc/slope
#     channel_unc_rel = ch_err/ch
#     peak_err_E = np.sqrt((slope*ch)**2 * ((slope_unc_rel)**2 + (channel_unc_rel)**2) + intercept_unc**2)
    
#     return peak_E, peak_err_E

def thompson(angle):
    angle = np.radians(angle)
    val = r0**2 * (1+(np.cos(angle))**2)/2
    return val

def klein_nishina(angle, gamma):
    val = thompson(angle)
    angle = np.radians(angle)
    val2 = 1/(1+gamma*(1-np.cos(angle)))**2
    val3 = 1 + (gamma**2 * (1-np.cos(angle))**2) / ((1+(np.cos(angle))**2) * (1+gamma*(1-np.cos(angle))))
    return val * val2 * val3
    
# def fit_peak(ch, counts, guess, half_width, offset=True, sigma_y=None):
#     """
#     ch, counts : full spectrum
#     guess      : rough peak channel (anywhere near the top is fine)
#     half_width : window is guess +- half_width channels; use >= ~1.3 x FWHM
#     offset     : also fit a constant background
#     sigma_y    : optional per-channel errors for the FULL spectrum (see module docstring)
 
#     Returns a dict with amp, mu, mu_err, fwhm, fwhm_err, popt, x, y, model.
#     """
#     sel = np.abs(ch - guess) <= half_width
#     x, y = ch[sel], counts[sel]
 
#     # start values taken from the data, not from the guess
#     smooth = gaussian_filter1d(y, 5, mode="nearest")
#     amp0, mu0 = smooth.max(), x[smooth.argmax()]
#     sig0 = max(np.count_nonzero(smooth > amp0 / 2) / FWHM_PER_SIGMA, 3.0)
 
#     p0 = [amp0, mu0, sig0] + ([0.0] if offset else [])
#     lower = [0.0, x[0], 1.0] + ([-np.inf] if offset else [])
#     upper = [10 * y.max(), x[-1], x[-1] - x[0]] + ([np.inf] if offset else [])
 
#     popt, pcov = curve_fit(
#         gauss, x, y, p0=p0, bounds=(lower, upper),
#         sigma=None if sigma_y is None else np.asarray(sigma_y)[sel],
#         absolute_sigma=sigma_y is not None, max_nfev=20000,
#     )
#     perr = np.sqrt(np.diag(pcov))

#     # yield of peak
#     yield_exact = popt[0] * popt[2] * np.sqrt(2*np.pi)
#     yield_exact_err = (perr[0]/popt[0] + perr[2]/popt[2])*yield_exact
#     print(yield_exact, yield_exact_err)

#     return {
#         "amp": popt[0], "mu": popt[1], "mu_err": perr[1],
#         "fwhm": FWHM_PER_SIGMA * popt[2], "fwhm_err": FWHM_PER_SIGMA * perr[2],
#         "popt": popt, "x": x, "y": y, "model": lambda xx: gauss(xx, *popt),
#         "yield": (yield_exact, yield_exact_err),
#     }

def main():
    # angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
    angles = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    peak_energies, energy_errs, energy_errs_indep, yield_values = [], [], [], []
    plot_configs = {}

    for (angle, (centroid_guess, hw)) in PEAKS.items():
        subtracted_file = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + str(angle) + ".txt"
        print(f"Loading " + subtracted_file)
        ch, counts = load_spectrum(subtracted_file)

        # convert to energies
        energies = []
        for val in ch:
            energies.append(calib(val))

        energies = np.array(energies, dtype=float)
        err = np.nan

        # perform the gaussian fitting to find the peak energy
        E_guess, fwhm_exp = expected_peak(angle)
        print(E_guess, fwhm_exp)
        r = fit_peak_bounded(energies, counts, E_guess, fwhm_exp, bin_width=coeffs["values"][0])
        
        # r = fit_peak(ch, counts, centroid_guess, hw)
            
        mean, err, fwhm, fwhm_err, popt, yield_exact = r["mu"], r["mu_err"], r["fwhm"], r["fwhm_err"], r["popt"], r["yield"]
        print("angle: ", str(angle))
        print("centroid (keV): ", str(mean))
        print("fwhm: ", fwhm)

        # uncertainty calculations
        # per-angle window/model systematic: refit with different settings, take the spread
        mus = [fit_peak_bounded(energies, counts, E_guess, fwhm_exp, 0.2407,
                                offset=o, hw_factor=h)["mu"]
            for h in (1.5, 2.0, 2.5) for o in (False, True)]
        sigma_win = np.std(mus, ddof=1)

        # sigma_stat = 1.2 * err            # my simulation: the fit's formal error is ~20% low
        sigma_cal  = coeffs["sys_err"]    # calibration residual RMS
        sigma_slope = mean * (slope_unc / slope)           # calibration slope uncertainty
        combined_error = quad_sum(err, sigma_win, sigma_cal, sigma_slope)
        
        peak_energies.append(mean)
        energy_errs.append(combined_error)
        energy_errs_indep.append(err) # used for the ODR fitting to find slope and y-intercept
        yield_values.append(yield_exact)
        
        # plotting the individual fits
        hw = 2.0 * fwhm_exp                          # the half-width the fit used (hw_factor = 2.0)
        view = np.abs(energies - E_guess) <= hw
        x, y = energies[view], counts[view]

        fig, ax1 = plt.subplots()
        ax1.step(x, y, where="mid", color="C7", alpha=0.7, linewidth=2, label="subtracted")

        mu, sigma = r["popt"][1], r["popt"][2]
        x_fit = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
        ax1.plot(x_fit, r["model"](x_fit), color="C3", label="fit")
        ax1.set_title("Cs-137 Subtracted counts with fitting for angle=" + str(angle))
        ax1.set_xlabel('Energy (keV)', fontsize=7)
        ax1.set_ylabel('Counts', fontsize=7)
        # ax1.axvspan(r["x"][0], r["x"][-1], color="C0", alpha=0.07, label="fit window")

        # store plot info in a dictionary to combine into one 3x3 plot for all angles and fits
        plot_configs[angle] = (x, y, x_fit, r["model"](x_fit))
        fig.savefig("plots/cs137_data_runs/cs137_angle_" + str(angle) + "_fitted.pdf", format="pdf", bbox_inches="tight")
        fig.clf()

    # # remove the 21-29 angle measurements to get a better fit
    # peak_energies = peak_energies[:2] + peak_energies[11:]
    # energy_errs = energy_errs[:2] + energy_errs[11:]
    # angles = angles[:2] + angles[11:]

    peak_energies = np.array(peak_energies, dtype=float)
    energy_errs = np.array(energy_errs[0], dtype=float)
    energy_errs_indep = np.array(energy_errs_indep, dtype=float)
    print(energy_errs)

    # print(peak_channels)
    # print(channel_errs)
    
    fig, axs = plt.subplots(3, 3)

    # plot the fits for each angle in one 3x3 plot
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
            ax.set_yscale("log", nonpositive="clip")

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
    y_errs_indep = (energy_errs_indep/peak_energies)*y

    # need to do a linear ODR fit since we have x and y errors
    sol = odr_fit(
        lambda x, beta: beta[0] + beta[1] * x,
        cos_x,
        y,
        beta0=[1.50, 1.96],
        weight_x=cos_x_errs**-2,
        weight_y=y_errs_indep**-2
    ) 

    m, b = sol.beta[1], sol.beta[0]
    m_err, b_err = sol.sd_beta[1], sol.sd_beta[0]

    print(m, b)

    e_mass = 1/m
    e_mass_unc = ((m_err/m)*m)/2
    print(f"mc^2 = {e_mass:.3f} +/- {e_mass_unc:.3f} MeV")

    g_energy = 1/b
    g_energy_unc = ((b_err/b)*b)/2
    print(f"energy of incoming photon = {g_energy:.3f} +/- {g_energy_unc:.3f} MeV")

    y_errs = (energy_errs/peak_energies)*y
    fig, ax1 = plt.subplots()
    h_data = ax1.scatter(cos_x, y, marker='o', label='data', zorder=2)
    h_err = ax1.errorbar(cos_x, y, yerr=y_errs, xerr=cos_x_errs, label='error', fmt='none', color='C7', zorder=1)
    h_fit, = ax1.plot(cos_x, m * cos_x + b, color='red', label=f'Best fit: y = {m:.2f}x + {b:.2f}', zorder=3)
    ax1.set_xlabel(r'$1-\cos(\theta)$')
    ax1.set_ylabel(r"1/E' $(\text{MeV}^{-1})$")
    ax1.grid(alpha=0.3)
    ax1.set_title('Cs-137 Compton energy shift')
    ax1.legend(loc="upper left", handles=[h_data, h_err, h_fit])
    fig.tight_layout()
    fig.savefig("plots/compton_energy_shift.pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    # calculate the differential cross section
    r = 48.5-28.5 # distance from target to detector
    h = 4 # "height" of target, unc = 0.5
    d = 0.75*2.54 # diameter of target
    p = 2.7 # density of aluminum (gm/cm^3)
    N0 = 6.02214076*10**23 # Avogadro's number
    A = 27 # atomic weight of aluminum
    Z = 13 #atomic number of aluminum
    crystal_area = np.pi*2.54**2
    dOmega = crystal_area/(r)**2
    print(dOmega)
    dOmega_err = 2*1/r * dOmega
    N = np.pi * (d/2)**2 * h * p * N0 * Z / A
    # print(N)
    N_err = 0.5/h * N

    # read off the peak/total ratio and absorption efficiency for the detector at each angle and calculate the scale factor to get the corrected yield
    # scale factors read off plots
    scale_factors = np.array([2.886, 2.564, 2.404, 2.077, 1.838, 1.696, 1.489, 1.400, 1.302], dtype=float)
    scale_factors_err = np.array([0.1016, 0.0842, 0.0763, 0.0608, 0.0503, 0.0446, 0.0366, 0.0333, 0.0298], dtype=float) 
    corr_yield, corr_yield_err = [], []
    for i in range(len(scale_factors)):
        val = yield_values[i][0] * scale_factors[i]
        # val = yield_values[i][0]
        err = (yield_values[i][1]/yield_values[i][0] + scale_factors_err[i]/scale_factors[i]) * val
        corr_yield.append(val)
        corr_yield_err.append(err)

    corr_yield, corr_yield_err = np.array(corr_yield, dtype=float), np.array(corr_yield_err, dtype=float)
    print(yield_values[2])
    print(corr_yield, corr_yield_err)

    activity = 3.70*10**7 # given
    I0 = activity / (4*np.pi) / (48.5)**2
    print(I0)
    I0_err = 2*0.5/48.5 * I0
    
    cross_sections = corr_yield / (dOmega * N * I0 * 300) * 10**27
    cross_sections_err = []
    for i in range(len(cross_sections)):
        cross_sections_err.append((corr_yield_err[i]/corr_yield[i] + dOmega_err/dOmega + N_err/N + I0_err/I0) * cross_sections[i])
    cross_sections_err = np.array(cross_sections_err, dtype=float)
    print(cross_sections)
    print(cross_sections_err)
    print(yield_values[8][0])

    # calculate klein nishina and thompson values (not doing uncertainty calculations here)
    gamma = E0_KEV/MC2_KEV
    th_cross_sections = []
    kn_cross_sections = []
    for i in range(len(cross_sections)):
        th_cross_sections.append(thompson(x[i]) * 10**27)
        kn_cross_sections.append(klein_nishina(x[i], gamma) * 10**27)

    # plot the cross section over angle and include Thompson and Klein-Nishina predictions
    plt.plot(x, cross_sections, label='data', zorder=2)
    plt.plot(x, th_cross_sections, label='Thompson', color='C1')
    plt.plot(x, kn_cross_sections, label='Klein-Nishina', color='C2')
    plt.errorbar(x, cross_sections, yerr=cross_sections_err, xerr=angle_err, label='error', fmt='none', color='C7', zorder=1)
    plt.title("Cross section over angle")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel(r'$d\sigma/d\Omega (10^{-27} (\mathrm{cm}^2/\mathrm{sr})$')
    plt.legend()
    fig.savefig("cross_section_over_angle.pdf", format="pdf", bbox_inches="tight")

    # print table of yields and comparisons with theory
    print("angle  yield    err    SF  corr yield  xsec   KN    ratio")
    for a, (y, ye), sf, k in zip(x, yield_values, scale_factors, kn_cross_sections):   # kn_vals in 1e-27 cm2/sr
        xs = y * sf / (dOmega * N * I0 * 300) * 1e27
        print(f"{a:5d} {y:7.0f} {ye:6.0f} {sf:5.2f} {y*sf:7.0f} {xs:7.1f} {k:6.1f} {xs/k:6.2f}")


if __name__ == "__main__":
    main()



