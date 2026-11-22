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
from fit_peak_coarse import fit_peak_coarse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from odrpack import odr_fit

FWHM_PER_SIGMA = 2.3548
LOG_Y = False
angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
# angles = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
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

FWHM_662 = 135.5336128634135*coeffs["values"][0]+coeffs["values"][1]       # keV, FWHM of the Cs-137 peak in your calibration spectrum
print(FWHM_662)

def compton_energy(angle_deg, E0=E0_KEV):
    return E0 / (1.0 + (E0 / MC2_KEV) * (1.0 - np.cos(np.radians(angle_deg))))

def expected_peak(angle_deg):
    """(expected energy, expected FWHM) in keV; NaI-type resolution scales as sqrt(E)."""
    E = compton_energy(angle_deg)
    return E, FWHM_662 * np.sqrt(E / E0_KEV)

def gauss(x, amp, mu, sigma, offset=0.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

def fit_peak_bounded(x, counts, guess, fwhm_exp, bin_width, offset=True, sigma_y=None,
                     mu_tol=20.0, width_range=(0.6, 1.6), hw_factor=4.0,
                     double=False, direct=None, fit_direct=False):
    x = np.asarray(x, float)
    counts = np.asarray(counts, float)
    sig_exp = fwhm_exp / FWHM_PER_SIGMA
    scale = np.sqrt(2 * np.pi) / bin_width               # (counts/bin x keV) -> counts

    if double:
        d = direct or {"mu": E0_KEV, "sigma": FWHM_662 / FWHM_PER_SIGMA}
        mu2, s2 = float(d["mu"]), float(d["sigma"])
        fwhm2 = FWHM_PER_SIGMA * s2
        lo = min(guess - hw_factor * fwhm_exp, mu2 - hw_factor * fwhm2)
        hi = max(guess + hw_factor * fwhm_exp, mu2 + hw_factor * fwhm2)
    else:
        lo, hi = guess - hw_factor * fwhm_exp, guess + hw_factor * fwhm_exp

    sel = (x >= lo) & (x <= hi)
    xs, ys = x[sel], counts[sel]
    # if len(xs) < 10:
    #     raise ValueError(f"only {len(xs)} points in the fit window; spectrum is too coarsely binned")
    sy = None if sigma_y is None else np.asarray(sigma_y, float)[sel]
    base = float(np.median(ys))

    def height(g, f):                                    # peak height above the typical level
        near = np.abs(xs - g) < f / 2
        return max((ys[near].max() if near.any() else ys.max()) - base, 1e-3)

    if not double:
        p0 = [height(guess, fwhm_exp), guess, sig_exp] + ([base] if offset else [])
        lower = [0.0, guess - mu_tol, width_range[0] * sig_exp] + ([-np.inf] if offset else [])
        upper = [np.inf, guess + mu_tol, width_range[1] * sig_exp] + ([np.inf] if offset else [])
        popt, pcov = curve_fit(gauss, xs, ys, p0=p0, bounds=(lower, upper), sigma=sy,
                               absolute_sigma=sy is not None, max_nfev=20000)
        perr = np.sqrt(np.diag(pcov))
        J = np.array([popt[2], 0.0, popt[0]]) * scale
        return {
            "yield": popt[0] * popt[2] * scale, "yield_err": np.sqrt(J @ pcov[:3, :3] @ J),
            "mu": popt[1], "mu_err": perr[1],
            "fwhm": FWHM_PER_SIGMA * popt[2], "fwhm_err": FWHM_PER_SIGMA * perr[2],
            "popt": popt, "x": xs, "y": ys,
            "model": lambda xx: gauss(np.asarray(xx, float), *popt),
        }

    # double: theta = [A1, mu1, A2, offset] (+ [mu2, s2] if fit_direct); s1 = s2*sqrt(mu1/mu2)
    def unpack(theta):
        a1, mu1, a2, c = theta[:4]
        m2, sg2 = (theta[4], theta[5]) if fit_direct else (mu2, s2)
        return a1, mu1, a2, c, m2, sg2, sg2 * np.sqrt(mu1 / m2)

    def model(xx, *theta):
        a1, mu1, a2, c, m2, sg2, s1 = unpack(theta)
        return gauss(xx, a1, mu1, s1) + gauss(xx, a2, m2, sg2) + c

    p0 = [height(guess, fwhm_exp), guess, height(mu2, fwhm2), base if offset else 0.0]
    lower = [0.0, guess - mu_tol, 0.0, -np.inf]
    upper = [np.inf, guess + mu_tol, np.inf, np.inf]
    if not offset:
        lower[3], upper[3], p0[3] = -1e-9, 1e-9, 0.0
    if fit_direct:                                       # let the direct peak float
        p0 += [mu2, s2]
        lower += [mu2 - mu_tol, width_range[0] * s2]
        upper += [mu2 + mu_tol, width_range[1] * s2]
    popt, pcov = curve_fit(model, xs, ys, p0=p0, bounds=(lower, upper), sigma=sy,
                           absolute_sigma=sy is not None, max_nfev=20000)
    a1, mu1, a2, c, m2, sg2, s1 = unpack(popt)

    def derived(theta):
        a1_, mu1_, a2_, _, m2_, sg2_, s1_ = unpack(theta)
        return np.array([a1_ * s1_ * scale, a2_ * sg2_ * scale, s1_])

    def err_of(k):                                       # delta method, numerical gradient
        g = np.zeros(len(popt))
        for j in range(len(popt)):
            h = 1e-6 * max(abs(popt[j]), 1.0)
            tp, tm = popt.copy(), popt.copy()
            tp[j] += h
            tm[j] -= h
            g[j] = (derived(tp)[k] - derived(tm)[k]) / (2 * h)
        return float(np.sqrt(max(g @ pcov @ g, 0.0)))

    perr = np.sqrt(np.diag(pcov))
    return {
        "yield": a1 * s1 * scale, "yield_err": err_of(0),
        "yield_direct": a2 * sg2 * scale, "yield_direct_err": err_of(1),
        "mu": mu1, "mu_err": perr[1],
        "fwhm": FWHM_PER_SIGMA * s1, "fwhm_err": FWHM_PER_SIGMA * err_of(2),  # tied, not fitted
        "mu_direct": m2, "fwhm_direct": FWHM_PER_SIGMA * sg2,
        "popt": np.array([a1, mu1, s1, a2, m2, sg2, c]), "x": xs, "y": ys,
        "model": lambda xx: model(np.asarray(xx, float), *popt),
        "compton_curve": lambda xx: gauss(np.asarray(xx, float), a1, mu1, s1),
        "direct_curve": lambda xx: gauss(np.asarray(xx, float), a2, m2, sg2),
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
    
def fit_peak(x, counts,  guess, fwhm_exp, bin_width, offset=False, sigma_y=None, hw_factor=2.0):
    """
    ch, counts : full spectrum
    guess      : rough peak channel (anywhere near the top is fine)
    half_width : window is guess +- half_width channels; use >= ~hw_factor x FWHM
    offset     : also fit a constant background
    sigma_y    : optional per-channel errors for the FULL spectrum (see module docstring)
 
    Returns a dict with amp, mu, mu_err, fwhm, fwhm_err, popt, x, y, model.
    """
    x = np.asarray(x, float)
    counts = np.asarray(counts, float)
    sel = np.abs(x - guess) <= hw_factor * fwhm_exp

    xs, ys = x[sel], counts[sel]
    # sig_exp = fwhm_exp / FWHM_PER_SIGMA
 
    # start values taken from the data, not from the guess
    smooth = gaussian_filter1d(ys, 5, mode="nearest")
    amp0, mu0 = smooth.max(), xs[smooth.argmax()]
    sig0 = max(np.count_nonzero(smooth > amp0 / 2) / FWHM_PER_SIGMA, 3.0)
 
    p0 = [amp0, mu0, sig0] + ([0.0] if offset else [])
    lower = [0.0, xs[0], 1.0] + ([-np.inf] if offset else [])
    upper = [10 * ys.max(), xs[-1], xs[-1] - xs[0]] + ([np.inf] if offset else [])
 
    popt, pcov = curve_fit(
        gauss, xs, ys, p0=p0, bounds=(lower, upper),
        sigma=None if sigma_y is None else np.asarray(sigma_y)[sel],
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

def main():
    angles = ["10", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "40", "50", "60", "70", "80", "90"]
    # angles = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    peak_energies, energy_errs, energy_errs_indep, yield_values, yield_errs = [], [], [], [], []
    plot_configs = {}

    subtracted_file = "Grad26/cs137_subtracted/cs137_subtracted_angle_0.txt"
    print(f"Loading " + subtracted_file)
    ch0, counts0 = load_spectrum(subtracted_file)

    # convert to energies
    energies0 = []
    for val in ch0:
        energies0.append(calib(val))
    
    energies0 = np.array(energies0, dtype=float)
    err = np.nan
    
    # perform the gaussian fitting to find the peak energy
    r0 = fit_peak_bounded(energies0, counts0, 661.657, FWHM_662, np.median(np.diff(energies0)))
    direct_peak = {"mu": r0["mu"], "sigma": r0["fwhm"] / 2.3548}
    MANUAL_OFFSET = {30: 0, 40: 150, 50: 150, 60: 100, 70: 100, 80: 70, 90: 120}   # your values


    for (angle, (centroid_guess, hw)) in PEAKS.items():
        subtracted_file = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + str(angle) + ".txt"
        print(f"Loading " + subtracted_file)
        ch, counts = load_spectrum(subtracted_file)

        # load no target file
        subtracted_nt_file = "Grad26/cs137_subtracted/cs137_subtracted_angle_" + str(angle) + "_no_target.txt"
        print(f"Loading " + subtracted_nt_file)
        ch_nt, counts_nt = load_spectrum(subtracted_nt_file)

        # convert to energies
        energies, energies_nt = [], []
        for val in ch:
            energies.append(calib(val))
        for val in ch_nt:
            energies_nt.append(calib(val))

        energies = np.array(energies, dtype=float)
        energies_nt = np.array(energies_nt, dtype=float)
        err = np.nan

        # perform the gaussian fitting to find the peak energy
        E_guess, fwhm_exp = expected_peak(angle)
        print(E_guess, fwhm_exp)
        bin_width = np.median(np.diff(energies))
        use_double = 10 <= angle <= 30
        # baseline (counts per bin) you read off each plot, next to the peak
        MANUAL_OFFSET = {30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0}   # your values
        # # MANUAL_OFFSET = {20: 0, 30: 0, 40: 0, 50: 0, 60: 0, 70: 0, 80: 0, 90: 0}   # your values

        off = MANUAL_OFFSET.get(angle)
        # if off is None:
        #     r = fit_peak_bounded(energies, counts, E_guess, fwhm_exp, bin_width,
        #                         double=use_double, direct=direct_peak if use_double else None,
        #                         fit_direct=use_double)
        # else:
        if 10 <= angle <= 30:
            r = fit_peak_bounded(energies, counts, E_guess, fwhm_exp + 5, bin_width,
                                    double=use_double, direct=direct_peak if use_double else None,
                                    fit_direct=use_double)
        else:
            r = fit_peak_bounded(energies, counts, E_guess, fwhm_exp + 25, bin_width, offset=False,
                                                double=use_double, direct=direct_peak if use_double else None,
                                                fit_direct=use_double)


        # fit the no target runs



        # r = fit_peak(energies, counts, E_guess, fwhm_exp, bin_width=coeffs["values"][0])

        
        # r = fit_peak(ch, counts, centroid_guess, hw)

        # bin_width = np.median(np.diff(energies))                 # keV per rebinned bin
        # sigma_E = direct_peak["sigma"] * np.sqrt(E_guess / 661.657)   # resolution at this energy
        # if angle >= 10:
        #     r = fit_peak_coarse(energies, counts, E_guess, fwhm_exp, bin_width, sigma=sigma_E, hw_factor=4.0)
        # else:
        #     r = fit_peak_coarse(energies, counts, E_guess, fwhm_exp, bin_width, sigma=sigma_E)

            
        mean, err, popt, yield_exact, yield_err = r["mu"], r["mu_err"], r["popt"], r["yield"], r["yield_err"]
        print("angle: ", str(angle))
        print("centroid (keV): ", str(mean))
        print("E guess: ", str(E_guess))
        # print("fwhm: ", fwhm)

        # uncertainty calculations
        # per-angle window/model systematic: refit with different settings, take the spread
        # mus = [fit_peak_bounded(energies, counts, E_guess, fwhm_exp, bin_width=coeffs["values"][0],
        #                         offset=o, hw_factor=h)["mu"]
        #     for h in (1.5, 2.0, 2.5) for o in (False, True)]
        # mus = [fit_peak(energies, counts, E_guess, fwhm_exp, bin_width=coeffs["values"][0], hw_factor=h)["mu"]
        #             for h in (1.5, 2.0, 2.5) for o in (False, True)]
        # sigma_win = np.std(mus, ddof=1)

        # sigma_stat = 1.2 * err            # my simulation: the fit's formal error is ~20% low
        sigma_cal  = coeffs["sys_err"]    # calibration residual RMS
        sigma_slope = mean * (slope_unc / slope)           # calibration slope uncertainty
        combined_error = quad_sum(err, sigma_cal, sigma_slope)
        print(err, sigma_cal, sigma_slope)
        print(combined_error)
        
        peak_energies.append(mean)
        energy_errs.append(combined_error)
        energy_errs_indep.append(err) # used for the ODR fitting to find slope and y-intercept
        yield_values.append(yield_exact)
        yield_errs.append(yield_err)
        
        # plotting the individual fits
        hw = 2.0 * fwhm_exp                          # the half-width the fit used (hw_factor = 2.0)
        view = np.abs(energies - E_guess) <= 1.3 * hw
        # x, y = energies[view], counts[view]
        x, y = energies, counts

        fig, ax1 = plt.subplots()
        ax1.step(x, y, color="C7", alpha=0.7, linewidth=2, label="subtracted")

        mu, sigma = r["popt"][1], r["popt"][2]
        # x_fit = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
        x_fit = np.linspace(energies, 400)
        y_fit = r["model"](x_fit) + (off or 0)
        ax1.plot(x_fit, y_fit, color="C3", label="fit")
        ax1.set_title("Cs-137 Subtracted counts with fitting for angle=" + str(angle))
        ax1.set_xlabel('Energy (keV)', fontsize=7)
        ax1.set_ylabel('Counts', fontsize=7)
        # ax1.axvspan(r["x"][0], r["x"][-1], color="C0", alpha=0.07, label="fit window")

        # store plot info in a dictionary to combine into one 3x3 plot for all angles and fits
        plot_configs[angle] = (x, y, x_fit, y_fit)
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
    
    # plot the fits for each angle in two 3x3 plots

    GRID_MAIN = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    GRID_FINE = [21, 22, 23, 24, 25, 26, 27, 28, 29]

    fig,  axs  = plt.subplots(3, 3)
    fig1, axs1 = plt.subplots(3, 3)

    for angle, (x, y, x_fit, y_fit) in plot_configs.items():
        a = int(angle)
        if a in GRID_MAIN:
            r, c = divmod(GRID_MAIN.index(a), 3)
            ax = axs[r, c]
        elif a in GRID_FINE:
            r, c = divmod(GRID_FINE.index(a), 3)
            ax = axs1[r, c]
        else:
            continue

        ax.step(x, y, color='C7', alpha=0.7, linewidth=2, label='background subtracted')
        ax.plot(x_fit, y_fit, color='C3', label='fit')
        ax.set_title(f'Angle = {a} deg', fontsize=9)
    
    fig.tight_layout()
    fig.savefig("plots/cs137_data_runs/cs137_subtracted_counts_w_fitting.pdf", format="pdf", bbox_inches="tight")

    fig1.tight_layout()
    fig1.savefig("plots/cs137_data_runs/cs137_subtracted_counts_21_29_w_fitting.pdf", format="pdf", bbox_inches="tight")

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
    dOmega_err = 2*1/r * dOmega
    N = np.pi * (d/2)**2 * h * p * N0 * Z / A
    # print(N)
    N_err = 0.5/h * N

    # read off the peak/total ratio and absorption efficiency for the detector at each angle and calculate the scale factor to get the corrected yield
    # scale factors read off plots
    # scale_factors = np.array([2.886, 2.564, 2.404, 2.077, 1.838, 1.696, 1.489, 1.400, 1.302], dtype=float)
    scale_factors = np.array([2.886, 2.564, 2.564, 2.514, 2.514, 2.514, 2.482, 2.482, 2.434, 2.434, 2.434, 2.404, 2.077, 1.838, 1.696, 1.489, 1.400, 1.302], dtype=float)
    # scale_factors_err = np.array([0.1016, 0.0842, 0.0763, 0.0608, 0.0503, 0.0446, 0.0366, 0.0333, 0.0298], dtype=float) 
    scale_factors_err = np.array([0.3465, 0.39, 0.39, 0.3978, 0.3978, 0.3978, 0.4029, 0.4029, 0.4108, 0.4108, 0.4108, 0.416, 0.4814, 0.544, 0.5896, 0.6716, 0.7144, 0.768], dtype=float) 
    corr_yield, corr_yield_err = [], []

    print(yield_values)
    print(yield_errs)
    for i in range(len(scale_factors)):
        val = yield_values[i] * scale_factors[i]
        # val = yield_values[i]
        err = (yield_errs[i]/yield_values[i] + scale_factors_err[i]/scale_factors[i]) * val
        corr_yield.append(val)
        corr_yield_err.append(err)

    corr_yield, corr_yield_err = np.array(corr_yield, dtype=float), np.array(corr_yield_err, dtype=float)
    print(corr_yield, corr_yield_err)

    activity = 3.70*10**7 # given
    I0 = activity / (4*np.pi) / (48.5)**2
    I0_err = 2*0.5/48.5 * I0
    
    cross_sections = corr_yield / (dOmega * N * I0 * 300) * 10**27
    cross_sections_err = []
    for i in range(len(cross_sections)):
        cross_sections_err.append((corr_yield_err[i]/corr_yield[i] + dOmega_err/dOmega + N_err/N + I0_err/I0) * cross_sections[i])
    cross_sections_err = np.array(cross_sections_err, dtype=float)
    print(cross_sections)
    print(cross_sections_err)

    # calculate klein nishina and thompson values (not doing uncertainty calculations here)
    gamma = E0_KEV/MC2_KEV
    th_cross_sections = []
    kn_cross_sections = []
    # x_theory = np.arange(0, 190, 10, dtype=int)
    for i in range(len(x)):
        th_cross_sections.append(thompson(x[i]) * 10**27)
        kn_cross_sections.append(klein_nishina(x[i], gamma) * 10**27)

    # plot the cross section over angle and include Thompson and Klein-Nishina predictions
    plt.plot(x[1:], cross_sections[1:], label='data', zorder=2)
    plt.plot(x, th_cross_sections, label='Thompson', color='C1')
    plt.plot(x, kn_cross_sections, label='Klein-Nishina', color='C2')
    plt.errorbar(x[1:], cross_sections[1:], yerr=cross_sections_err[1:], xerr=angle_err, label='error', fmt='none', color='C7', zorder=1)
    plt.title("Cross section over angle")
    plt.xlabel("Scattering angle (deg)")
    plt.ylabel(r'$d\sigma/d\Omega (10^{-27} (\mathrm{cm}^2/\mathrm{sr})$')
    plt.legend()
    fig.savefig("plots/cross_section_over_angle.pdf", format="pdf", bbox_inches="tight")

    # print table of yields and comparisons with theory
    print("angle  yield    err    SF  corr yield  xsec   KN    ratio")
    for a, y, ye, sf, k in zip(x, yield_values, yield_errs, scale_factors, kn_cross_sections):   # kn_vals in 1e-27 cm2/sr
        xs = y * sf / (dOmega * N * I0 * 300) * 1e27
        print(f"{a:5d} {y:7.0f} {ye:6.0f} {sf:5.2f} {y*sf:7.0f} {xs:7.1f} {k:6.1f} {xs/k:6.2f}")


if __name__ == "__main__":
    main()



