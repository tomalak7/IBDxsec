"""
Energy Spectra and Radiative Corrections in Inverse Beta Decay (IBD)
---------------------------------------------------------------------

This script computes the energy and angular spectra in inverse beta decay
for a variety of next-to-leading order corrections, including
- QED radiative corrections (virtual, real soft and hard, region-specific)
- Recoil, weak magnetism, and nucleon structure corrections

The script provides results with sub-permille accuracy for antineutrino energies below 10 MeV.

Electromagnetic energy: the sum of the positron and photon energies.

Key Functions:
---------------
- polylog2(z):           Polylogarithm of order 2, implemented via the Spence function.
- recoil(...):           Recoil correction.
- WM(...):               Weak magnetism correction.
- FF(...):               Nucleon structure (form factor) correction.
- result_LO(...):        Leading-order IBD spectrum with recoil, weak magnetism, and nucleon structure corrections.
- delta_v(...):          Factorizable virtual-photon QED correction.
- f2(...):               QED form factor f2 for nonfactorizable virtual-photon QED correction.
- delta_s(...):          Soft-photon QED correction.
- delta_2_em(...):       Region 2 correction for electromagnetic energy spectrum.
- delta_2p(...):         Region 2 correction for positron energy spectrum.
- delta_1(...):          Outer radiative correction of the angular-independent terms.
- delta_2(...):          Outer radiative correction of the angular-dependent terms.
- result_EM_static(...): QED radiative correction to electromagnetic energy spectrum with phase-space integration in the static limit.
- result_EM(...):        QED radiative correction to electromagnetic energy spectrum.
- result_elastic(...):   QED radiative correction to positron energy spectrum in elastic kinematics.
- result_inelastic(...): Positron energy spectrum in inelastic (radiative) kinematics.
- result2D(...):         2D distribution in positron energy and angle in radiative IBD.

Usage:
------
When run as a script, the code fetches particle masses and physical constants from SciPy,
sets up example kinematics, and prints out the computed spectra for demonstration.

Functions can be used for further evaluations only after additional tests.

Dependencies:
-------------
- numpy
- scipy.special.spence

Author: Oleksandr Tomalak
Date: 12-07-2025

References:
-----------
- Tomalak, O. (2025). Theory of inverse beta decay for reactor antineutrinos, arXiv:2512.07956
"""

import numpy as np
from scipy import constants
from scipy.special import spence

def polylog2(z):
    """
    Compute the polylogarithm of order 2 (dilogarithm), PolyLog[2, z].
    Implemented via the Spence function: PolyLog[2, z] = -spence(1 - z).
    Parameters
    ----------
    z : float or np.ndarray
        Argument of the polylogarithm.
    Returns
    -------
    float or np.ndarray
        PolyLog[2, z].
    """
    # Add a small imaginary part to avoid a branch cut for real z > 1
    return spence(1 - z + 1e-12j).real

def recoil(mp, mn, me, gV, gA, E0, Enu, Q2):
    """
    Compute the recoil correction.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    E0 : float
        Maximum electron energy in neutron decay (GeV).
    Enu : float
        Antineutrino energy (GeV).
    Q2 : float
        Squared 4-momentum transfer (GeV²).
    Returns
    -------
    float
        Recoil correction (10⁻⁴² cm²/GeV).
    """
    term1 = -(E0 / Enu) * (gV**2 + gA**2)
    term2 = - (gV**2 - gA**2) * (Q2 + me**2) / (4 * Enu**2)
    term3 = ((Q2 + me**2) / (4 * Enu**2) - Q2 / (2 * Enu * E0)) * (gV - gA)**2
    return (mp / np.pi) * (E0 / mn) * (term1 + term2 + term3)

def WM(mp, mn, me, gV, gA, muV, E0, Enu, Q2):
    """
    Compute the weak magnetism correction.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (in GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    muV : float
        Isovector-vector magnetic moment.
    E0 : float
        Maximum electron energy in neutron decay (GeV).
    Enu : float
        Antineutrino energy (GeV).
    Q2 : float
        Squared 4-momentum transfer (GeV²).
    Returns
    -------
    float
        Weak magnetism correction (10⁻⁴² cm²/GeV).
    """
    term1 = E0 / mn * (Q2 + me**2) / (2 * Enu**2)
    term2 = - Q2 / (mn * Enu)
    return (mp / np.pi) * (term1 + term2) * gA * (muV - 1)

def FF(mp, me, gV, gA, rV2, rA2, E0, Enu, Q2, hc):
    """
    Compute the nucleon structure (form factor) correction.
    Parameters
    ----------
    mp, me : float
        Masses of proton and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    E0 : float
        Maximum electron energy in neutron decay (GeV).
    Enu : float
        Antineutrino energy (GeV).
    Q2 : float
        Squared 4-momentum transfer (GeV²).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    Returns
    -------
    float
        Nucleon structure correction (10⁻⁴² cm²/GeV).
    """
    term1 = (1 - E0 / Enu) * (gA**2 * rA2 + gV**2 * rV2)
    term2 = - (Q2 + me**2) / (4 * Enu**2) * (gV**2 * rV2 - gA**2 * rA2)
    return - hc**-2 * (mp / np.pi) * (term1 + term2) * Q2 / 3

def result_LO(mp, mn, me, gV, gA, rV2, rA2, muV, Enu, Ee, hc, GF, Vud):
    """
    Compute the leading-order positron spectrum for inverse beta decay, 
    including recoil, weak magnetism, and nucleon structure corrections.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    Enu : float
        Antineutrino energy (GeV).
    Ee : float
        Positron energy (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).⁻²
    Vud : float
        CKM matrix element.
    Returns
    -------
    float
        Differential cross section (10⁻⁴² cm²/GeV).
    """
    En = mp + Enu - Ee
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Q2 = mp**2 - mn**2 - 2 * mp * (mp - En)

    prefactor = 1e16 * hc**2 * GF**2 * Vud**2

    term1 = ((gV**2 + gA**2) * Ee / Enu - (me**2 + Q2) / (4 * Enu**2) * (gV**2 - gA**2)) * mp / np.pi
    term2 = recoil(mp, mn, me, gV, gA, E0, Enu, Q2)
    term3 = WM(mp, mn, me, gV, gA, muV, E0, Enu, Q2)
    term4 = FF(mp, me, gV, gA, rV2, rA2, E0, Enu, Q2, hc)

    return prefactor * (term1 + term2 + term3 + term4)

def delta_v(beta):
    """
    Virtual-photon QED correction.
    The renormalization scale is chosen as the electron mass.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    Returns
    -------
    float
        Virtual correction term.
    """
    log_term = np.log(np.abs((1 + beta) / (1 - beta)))
    return (
        -3 / 4 +
        (1 / (2 * beta)) * (
            polylog2( (1 + beta) / (2 * beta)) -
            polylog2( (beta - 1) / (2 * beta)) +
            (1 - np.log(np.abs((2 * beta) / (1 - beta)))) * log_term +
            0.5 * log_term**2 -
            np.pi**2 / 2
        )
    )

def f2(beta):
    """
    Form factor f2 for virtual-photon QED correction.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    Returns
    -------
    float
        QED form factor f2.
    """
    return (np.sqrt(1 - beta**2) / (4 * beta)) * np.log(np.abs((1 + beta) / (1 - beta)))

def delta_s(beta, me, epsilon):
    """
    Soft-photon QED correction.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    me : float
        Positron mass.
    epsilon : float
        Soft-photon energy cutoff.
    Returns
    -------
    float
        Soft correction term.
    """
    term1 = (1 / beta) * (polylog2( (1 - beta) / (1 + beta)) - np.pi**2 / 6)
    term2 = -(1 - (1 / (2 * beta)) * np.log(np.abs((1 + beta) / (1 - beta)))) * np.log(np.abs((4 * epsilon**2) / me**2))
    term3 = (1 / (2 * beta)) * np.log(np.abs((1 + beta) / (1 - beta))) * (1 + np.log(np.abs((np.sqrt(1 - beta**2) * (1 + beta)) / (4 * beta**2))))
    return term1 + term2 + term3 + 1

def delta_2_em(beta):
    """
    Region 2 correction for the electromagnetic energy spectrum.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    Returns
    -------
    float
        Region 2 correction for the electromagnetic energy spectrum.
    """
    return -(1 - (1 / (2 * beta)) * np.log(np.abs((1 + beta) / (1 - beta)))) * np.log(np.abs((1 + beta) / (1 - beta)))

def delta_2p(Enu, Ee, En, mn, me):
    """
    Region 2 correction for the positron energy spectrum.
    Parameters
    ----------
    Enu : float
        Antineutrino energy (GeV).
    Ee : float
        Positron energy (GeV).
    En : float
        Final-state neutron energy (GeV).
    mn : float
        Neutron mass (GeV).
    me : float
        Positron mass (GeV).
    Returns
    -------
    float
        Region 2 correction for the positron energy spectrum.
    """
    beta = np.sqrt(1 - me**2 / Ee**2)
    beta_x = np.sqrt(1 - mn**2 / En**2)
    rho = np.sqrt(1 - beta**2)
    rho_x = np.sqrt(1 - beta_x**2)

    cos_delta = (Enu**2 - beta**2 * Ee**2 - (En**2 - mn**2)) / (2 * beta * Ee * np.sqrt(En**2 - mn**2))
    
    A = 1 - beta * beta_x * cos_delta
    B = np.sqrt(A**2 - rho**2 * rho_x**2)

    log1 = np.log(np.abs((1 + beta_x) / (1 - beta_x)))
    log2 = np.log(np.abs((1 + beta) / (1 - beta)))

    term1 = -1 - (1 - 1 / (2 * beta_x)) * log1 - (1 / (2 * beta)) * log2
    term2 = (1 - beta * beta_x * cos_delta) / (2 * B) * np.log(np.abs((A + B) / (A - B)))

    L = np.log(np.abs((1 + beta) / (1 - beta)))
    P1 = polylog2( ((-1 + cos_delta) * (-1 + beta) * beta_x) / (beta - cos_delta * beta_x + B))
    P2 = polylog2( -(((-1 + cos_delta) * (1 + beta) * beta_x) / (beta - cos_delta * beta_x + B)))
    P3 = polylog2( -(((-1 + cos_delta) * (-1 + beta) * beta_x) / (-beta + cos_delta * beta_x + B)))
    P4 = polylog2( ((-1 + cos_delta) * (1 + beta) * beta_x) / (-beta + cos_delta * beta_x + B))

    term3 = (1 / beta) * (
        0.5 * L**2 +
        L * np.log(np.abs((2 * beta * (beta_x + 1)) / ((beta + 1) * beta_x * (cos_delta + 1)))) -
        P1 + P2 - P3 + P4
    )

    return term1 + term2 + term3

def delta_1(beta):
    """
    Outer radiative correction of the angular-independent terms.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    Returns
    -------
    float
        Outer radiative correction of the angular-independent terms.
    """
    log_term = np.log((1 + beta) / (1 - beta))
    
    term1 = 7/2
    term2 = (7 + 3 * beta**2) / (8 * beta) * log_term
    term3 = 2 * (1/(2 * beta) * log_term - 1) * np.log((4 * beta**2) / (1 - beta**2))
    term4 = -1/beta * log_term**2
    term5 = -4/beta * polylog2((2 * beta) / (1 + beta))
    
    return term1 + term2 + term3 + term4 + term5

def delta_2(beta):
    """
    Outer radiative correction of the angular-dependent terms.
    Parameters
    ----------
    beta : float
        Velocity parameter for the positron (v/c).
    Returns
    -------
    float
        Outer radiative correction of the angular-dependent terms.
    """
    log_term = np.log((1 + beta) / (1 - beta))
    sqrt_1pbeta = np.sqrt(1 + beta)
    sqrt_1mbeta = np.sqrt(1 - beta)
    
    term1 = 1
    term2 = (2 * (1 - np.sqrt(1 - beta**2))) / beta**2
    term3 = -(1 - 1/(4 * beta)) * log_term
    term4 = -2 * (1/(2 * beta) * log_term - 1) * np.log(
        0.5 * (1 + 1/beta) * (sqrt_1pbeta + sqrt_1mbeta) / (sqrt_1pbeta - sqrt_1mbeta)
    )
    term5 = 1/16 * (4/beta - 3 - 1/beta**2) * log_term**2
    term6 = -4/beta * polylog2(1 - sqrt_1mbeta/sqrt_1pbeta)
    
    return term1 + term2 + term3 + term4 + term5 + term6

def result_EM_static(mp, mn, me, gV, gA, rV2, rA2, muV, Enu, Ee, epsilon, hc, GF, Vud, alpha, delta_1, delta_2):
    """
    Radiative correction to electromagnetic energy spectrum with phase-space integration in the static limit.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    Enu : float
        Antineutrino energy (GeV).
    Ee : float
        Electromagnetic energy (GeV).
    epsilon : float
        Soft-photon energy cutoff (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).
    Vud : float
        CKM matrix element.
    alpha : float
        Fine-structure constant.
    delta_1, delta_2 : callable
        Outer radiative correction2.
    Returns
    -------
    float
        Electromagnetic energy spectrum (10⁻⁴² cm²/GeV).
    """
    En = mp + Enu - Ee
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Q2 = mp**2 - mn**2 - 2 * mp * (mp - En)
    beta0 = np.sqrt(1 - me**2 / Ee**2)
    
    prefactor = 1e16 * hc**2 * alpha / np.pi * GF**2 * Vud**2
    main_term = mp / np.pi * np.real(
        ((gV**2 + gA**2) * Ee / Enu - (me**2 + Q2) / (4 * Enu**2) * (gV**2 - gA**2)) *  delta_1(beta0) - 
        (me**2 + Q2) / (8 * Enu**2) * (gV**2 - gA**2) * (delta_2(beta0) - 3.*delta_1(beta0))
    )
    
    return prefactor * main_term

def result_EM(mp, mn, me, gV, gA, rV2, rA2, muV, Enu, Ee, epsilon, hc, GF, Vud, alpha,
              delta_v, f2, delta_s, delta_2EM):
    """
    Radiative correction to electromagnetic energy spectrum in IBD.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    Enu : float
        Antineutrino energy (GeV).
    Ee : float
        Electromagnetic energy (GeV).
    epsilon : float
        Soft-photon energy cutoff (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).
    Vud : float
        CKM matrix element.
    alpha : float
        Fine-structure constant.
    delta_v, f2, delta_s, delta_2EM : callable
        Functions for QED virtual, soft, and region 2.
    Returns
    -------
    float
        Electromagnetic energy spectrum (10⁻⁴² cm²/GeV).
    """
    En = mp + Enu - Ee
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Q2 = mp**2 - mn**2 - 2 * mp * (mp - En)
    beta = np.sqrt(1 - mn**2 / En**2)
    beta0 = np.sqrt(1 - me**2 / Ee**2)

    prefactor = 1e16 * hc**2 * alpha / np.pi * GF**2 * Vud**2
    pi_inv = mp / np.pi / Enu  # reuse multiple times

    term1 = (Ee / 4) * (me**2 / (Ee**2 - (Enu - En * beta)**2) - 1) * pi_inv * (gV**2 + gA**2)
    
    term2_factor = (me**2 / (Ee**2 - (Enu - beta * En)**2) - 1)
    term2a = term2_factor * (En**2 - mn**2 - Ee**2 - (Enu - Ee)**2 + (Enu - beta * En)**2) / (16 * Enu)
    term2b = -(Ee**2 - me**2 - (Enu - beta * En)**2) / (4 * Enu)
    term2 = -(term2a + term2b) * pi_inv * (gV**2 - gA**2)
    
    log_arg1 = (1 + beta0) / (1 - beta0)
    log_term = np.log(np.abs(log_arg1))
    
    term3 = (Ee * (gV**2 + gA**2) + (-(Ee / 2) + (1 - beta0) / beta0 * (En**2 - mn**2 - Enu**2 - Ee**2 + me**2) / (4 * Enu)) * (gV**2 - gA**2)) * pi_inv * log_term
    
    term4_factor = (-(Ee / 4) * (gA**2 + gV**2) + (2 * (En**2 - mn**2 - Enu**2 - Ee**2) - mn * E0 + mp * (Enu - Ee)) / (8 * Enu) * (gV**2 - gA**2))
    term4 = - term4_factor * pi_inv * np.log(np.abs((Ee**2 - (Enu - beta * En)**2) / me**2))
    
    term5 = (Ee / 2) * (gV**2 - gA**2) * pi_inv * np.log(np.abs((Ee + Enu - beta * En) / (Ee - Enu + beta * En)))
    
    delta_log_arg = (2 * me * epsilon) / (Ee**2 - (Enu - beta * En)**2 - me**2)
    
    polylog_sum = (
        polylog2(-1 + (2 * Ee) / me * np.sqrt((1 - beta0) / (1 + beta0))) -
        polylog2(-1 + (2 * Ee) / me * np.sqrt((1 + beta0) / (1 - beta0))) +
        polylog2((Ee - Enu + beta * En) / me * np.sqrt((1 + beta0) / (1 - beta0))) +
        polylog2((Ee + Enu - beta * En) / me * np.sqrt((1 + beta0) / (1 - beta0))) -
        polylog2((Ee + Enu - beta * En) / me * np.sqrt((1 - beta0) / (1 + beta0))) -
        polylog2((Ee - Enu + beta * En) / me * np.sqrt((1 - beta0) / (1 + beta0)))
    )
    
    log_ratio1 = np.log(np.abs((Ee + Enu - beta * En) / (Ee - Enu + beta * En)))
    log_ratio2 = np.log(np.abs((beta0 * Ee + Enu - beta * En) / (beta0 * Ee - Enu + beta * En)))
    
    big_log_term = (
        np.log(np.abs(log_arg1)) *
        (np.log(np.abs((Ee**2 - (Enu - beta * En)**2) / (4 * me**2))) +
         np.log(np.abs((Ee**2 - (Enu - beta * En)**2 - me**2) / (Ee**2 - me**2)))) +
        polylog_sum +
        log_ratio1 * log_ratio2 +
        0.5 * log_term**2
    )
    
    term6 = ((gV**2 + gA**2) * Ee / Enu - (me**2 + Q2) / (4 * Enu**2) * (gV**2 - gA**2)) * mp / np.pi * (
        delta_v(beta0) + delta_s(beta0, me, epsilon) + delta_2EM(beta0) +
        2 * (1 - 1 / (2 * beta0) * log_term) * np.log(np.abs(delta_log_arg)) -
        1 / beta0 * big_log_term
    )
    
    term7 = mp / np.pi * me / Enu * (gV**2 + 3 * gA**2) * f2(beta0)

    return prefactor * (term1 + term2 + term3 + term4 + term5 + term6 + term7)

def result_elastic(mp, mn, me, gV, gA, rV2, rA2, muV, Enu, Ee, epsilon, hc, GF, Vud, alpha,
                   delta_v, f2, delta_s, delta_2p):
    """
    Radiative correction to positron energy spectrum for elastic kinematics in IBD.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    Enu : float
        Antineutrino energy (GeV).
    Ee : float
        Positron energy (GeV).
    epsilon : float
        Soft-photon energy cutoff (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).
    Vud : float
        CKM matrix element.
    alpha : float
        Fine-structure constant.
    delta_v, f2, delta_s, delta_2p : callable
        Functions for QED virtual, soft, and region 2.
    Returns
    -------
    float
        Positron energy spectrum (10⁻⁴² cm²/GeV).
    """
    l0 = mp + Enu - Ee
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Delta = mp * Enu - mn * E0
    s = mp**2 + 2 * mp * Enu
    Sigma = (s - (me - mn)**2) * (s - (me + mn)**2)
    Q2 = mp**2 - mn**2 - 2 * mp * (Ee - Enu)
    beta = np.sqrt(1 - me**2 / Ee**2)

    prefactor = 1e16 * hc**2 * alpha / np.pi * GF**2 * Vud**2
    pi_inv = mp / np.pi

    # Break down terms for clarity
    term1 = pi_inv * (gV**2 - gA**2) * (l0**2 - mn**2 - (Enu - Ee * beta)**2) / (8 * Enu**2)

    term2 = pi_inv * (gV**2 - gA**2) * mn**2 / (8 * Enu**2) * np.log(np.abs((l0**2 - (Enu - Ee * beta)**2) / mn**2))

    term3 = -pi_inv * (gV**2 - gA**2) * Ee / (4 * Enu) * np.log(np.abs((1 + beta) / (1 - beta)))

    term4 = pi_inv * (gV**2 - gA**2) * (-me**2 + mn**2 + 2 * Ee * (l0 + Ee) - s) / (8 * beta * Enu**2) * np.log(np.abs((1 + beta) / (1 - beta)))

    sqrt_l0 = np.sqrt(l0**2 - mn**2)
    log1 = np.log(np.abs((l0 - sqrt_l0) / (l0 + sqrt_l0)))
    log2 = np.log(np.abs((l0 + (Enu - Ee * beta)) / (l0 - (Enu - Ee * beta))))
    log3 = np.log(np.abs((l0**2 - (Enu - Ee * beta)**2) / mn**2))

    term5 = pi_inv * ((gV**2 + gA**2) - (Enu - Ee) / (2 * Enu) * (gV**2 - gA**2)) * (
        sqrt_l0 / (2 * Enu) * log1 +
        (Enu - Ee * beta) / (2 * Enu) * log2 +
        l0 / (2 * Enu) * log3
    )

    term6_factor = ((gV**2 + gA**2) + mp / (2 * Enu) * (gV**2 - gA**2))
    term6 = -pi_inv * term6_factor * mn**2 / (s - me**2) * l0 / (2 * Enu) * np.log(np.abs((s + me**2 - 2 * Ee * (l0 + Ee - Enu * beta)) / mn**2))

    term7 = -pi_inv * term6_factor * Delta / (s - me**2) * (l0 + Ee) / (2 * Enu) * np.log(np.abs((Sigma - 4 * (Delta + me**2)**2) / (4 * me**4)))

    term8 = pi_inv * term6_factor * ((l0**2) - s - me**2) / (s - me**2) * Ee / (2 * Enu) * np.log(np.abs((Sigma - 4 * (Delta + me**2)**2) / (4 * me**2 * s)))

    # Complex big term with multiple logs
    log_argA = 1 - (s * np.sqrt(1 - beta)) / (me * mp * np.sqrt(1 + beta))
    log_argB = (16 * me**10) / (s * (np.sqrt(Sigma) - 2 * Delta)**4)
    log_argC = (Sigma - 4 * (Delta + me**2)**2) / (4 * me**4)
    log_argD = (np.sqrt(Sigma) - 2 * me * np.sqrt(s))**2 - 4 * (Delta + me**2)**2
    log_argE = (np.sqrt(Sigma) + 2 * me * np.sqrt(s))**2 - 4 * (Delta + me**2)**2
    log_argF = (np.sqrt(Sigma) + 2 * Delta) / (np.sqrt(Sigma) - 2 * Delta)

    term9_inner = (
        (s - me**2) / s * np.log(np.abs(log_argA)) +
        l0 / (2 * Enu) * np.log(np.abs(log_argB)) -
        Ee / (2 * Enu) * np.log(np.abs(log_argC)) +
        (s - me**2) / s * (l0 + Ee) / Enu * np.log(np.abs((np.sqrt(Sigma) - 2 * Delta) / (2 * me**2))) +
        (mp**2 + me**2 - 2 * mp * Ee) / (2 * mp * Enu) *
        np.log(np.abs((1 - (s * np.sqrt(1 - beta)) / (me * mp * np.sqrt(1 + beta))) *
               (1 - (mp * np.sqrt(1 + beta)) / (me * np.sqrt(1 - beta))))) +
        ((s - me**2) / (2 * me * np.sqrt(s)) * Ee / Enu - me / np.sqrt(s) * l0 / (2 * Enu)) *
        np.log(np.abs(log_argD / log_argE)) +
        (-4 * (Delta + me**2) * (l0 + Ee) / Enu +
         (s + me**2 + 2 * Delta) * (l0 + 2 * Ee) / Enu -
         2 * ((Delta + me**2) * (2 * Delta + me**2) + (Delta - me**2) * s) / np.sqrt(Sigma) * l0 / Enu -
         ((s - me**2) * (np.sqrt(Sigma) - 2 * Delta)) / np.sqrt(Sigma) * (l0 + Ee) / Enu
        ) * np.log(np.abs(log_argF)) / (np.sqrt(Sigma) + 2 * (Delta + me**2))
    )
    term9 = -0.5 * pi_inv * term6_factor * (s - me**2 - mn**2) / (s - me**2) * term9_inner

    poly_arg1 = s / (me * mp) * np.sqrt((1 - beta) / (1 + beta))
    poly_arg2 = (mp * np.sqrt(1 + beta)) / (me * np.sqrt(1 - beta))
    poly_arg3 = (Delta + me**2 - np.sqrt(Sigma) / 2) / me**2
    poly_arg4 = (Delta + me**2 + np.sqrt(Sigma) / 2) / me**2

    term10 = 0.5 * pi_inv * (
        Ee / Enu * (gV**2 + gA**2) - (Q2 - me**2) / (4 * Enu**2) * (gV**2 - gA**2)
    ) * (polylog2(poly_arg1) + polylog2(poly_arg2) - polylog2(poly_arg3) - polylog2(poly_arg4))

    term11 = 2 * pi_inv * (1 - 1 / (2 * beta) * np.log(np.abs((1 + beta) / (1 - beta)))) * (
        ((1 - beta) / 2 * Ee / Enu - (me**2 + Q2) / (4 * Enu**2)) * (gV**2 - gA**2) +
        ((gV**2 + gA**2) * Ee / Enu - (me**2 + Q2) / (4 * Enu**2) * (gV**2 - gA**2)) *
        np.log(np.abs((l0 + np.sqrt(l0**2 - mn**2)) / ((Enu - beta * Ee)**2 - l0**2 + mn**2) * 2 * epsilon))
    )

    term12 = pi_inv * ((gV**2 + gA**2) * Ee / Enu - (me**2 + Q2) / (4 * Enu**2) * (gV**2 - gA**2)) * (
        delta_v(beta) + delta_s(beta, me, epsilon) + delta_2p(Enu, Ee, l0, mn, me)
    )

    term13 = pi_inv * me / Enu * (gV**2 + 3 * gA**2) * f2(beta)

    return prefactor * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13)

def result_inelastic(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, hc, GF, Vud, alpha):
    """
    Positron energy spectrum for inelastic (radiative) kinematics in IBD.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    E_nu : float
        Antineutrino energy (GeV).
    E_e : float
        Positron energy (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).
    Vud : float
        CKM matrix element.
    alpha : float
        Fine-structure constant.
    Returns
    -------
    float
        Inelastic positron energy spectrum (10⁻⁴² cm²/GeV).
    """
    pi = np.pi
    
    l0 = mp + E_nu - E_e
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Delta = mp * E_nu - mn * E0
    s = mp**2 + 2 * mp * E_nu
    Sigma = (mp**2 + 2*mp*E_nu - (me - mn)**2) * (mp**2 + 2*mp*E_nu - (me + mn)**2)
    Q2 = mp**2 - mn**2 - 2 * mp * (E_e - E_nu)
    beta = np.sqrt(1 - (me**2)/(E_e**2))
    
    term1 = (mp/pi) * E_e/(2*E_nu) * beta * (gV**2 - gA**2)
    term2 = -(mp/pi) * E_e/(2*E_nu) * (gV**2 - gA**2) * np.log(np.abs((1 + beta)/(1 - beta)))
    term3 = (mp/pi) * (mn**2)/(8 * E_nu**2) * (gV**2 - gA**2) * np.log(np.abs((l0**2 - (E_nu - E_e*beta)**2)/(l0**2 - (E_nu + E_e*beta)**2)))
    
    factor4 = (gV**2 + gA**2) - (E_nu - E_e)/(2*E_nu) * (gV**2 - gA**2)
    term4_part1 = ((E_nu - l0 + E_e*beta)/(2*E_nu)) * np.log(np.abs((E_nu - l0 + E_e*beta)/(E_nu + l0 + E_e*beta)))
    term4_part2 = -((E_nu - l0 - E_e*beta)/(2*E_nu)) * np.log(np.abs((E_nu - l0 - E_e*beta)/(E_nu + l0 - E_e*beta)))
    term4_part3 = (l0/E_nu) * np.log(np.abs((E_nu + l0 - E_e*beta)/(E_nu + l0 + E_e*beta)))
    term4 = (mp/pi) * factor4 * (term4_part1 + term4_part2 + term4_part3)
    
    factor5 = ((gV**2 + gA**2) + (mp/(2*E_nu)) * (gV**2 - gA**2))
    term5 = (mp/pi) * (mn**2)/(s - me**2) * (l0/(2*E_nu)) * factor5 * np.log(np.abs((me**2 + s - 2*E_e*(E_e + l0 + E_nu*beta)) / (me**2 + s - 2*E_e*(E_e + l0 - E_nu*beta))))
    
    term6 = -(mp/pi) * ((s - me**2 - mn**2)/(2*s)) * factor5 * np.log(np.abs((1 - (s*np.sqrt(1-beta))/(me*mp*np.sqrt(1+beta))) / (1 - (s*np.sqrt(1+beta))/(me*mp*np.sqrt(1-beta)))))
    
    term7_factor = (mp**2 + me**2 - 2*mp*E_e)/(4*mp*E_nu)
    term7_log_arg = ((1 - (s*np.sqrt(1-beta))/(me*mp*np.sqrt(1+beta))) * (1 - (mp*np.sqrt(1+beta))/(me*np.sqrt(1-beta)))) / ((1 - (mp*np.sqrt(1-beta))/(me*np.sqrt(1+beta))) * (1 - (s*np.sqrt(1+beta))/(me*mp*np.sqrt(1-beta))))
    term7 = -(mp/pi) * term7_factor * ((s - me**2 - mn**2)/(s - me**2)) * factor5 * np.log(np.abs(term7_log_arg))
    
    poly_arg1 = (s*np.sqrt(1-beta)) / (me*mp*np.sqrt(1+beta))
    poly_arg2 = (mp*np.sqrt(1+beta)) / (me*np.sqrt(1-beta))
    poly_arg3 = (s*np.sqrt(1+beta)) / (me*mp*np.sqrt(1-beta))
    poly_arg4 = (mp*np.sqrt(1-beta)) / (me*np.sqrt(1+beta))
    
    poly_sum = polylog2(poly_arg1) + polylog2(poly_arg2) - polylog2(poly_arg3) - polylog2(poly_arg4)
    
    term8_factor = (E_e/E_nu)*(gV**2 + gA**2) - ((Q2 - me**2)/(4*E_nu**2))*(gV**2 - gA**2)
    term8 = (mp/pi) * term8_factor * 0.5 * poly_sum
    
    log_arg = ((E_nu + beta*E_e)**2 - l0**2 + mn**2) / ((E_nu - beta*E_e)**2 - l0**2 + mn**2)
    term9_factor = 1 - (1/(2*beta)) * np.log(np.abs((1+beta)/(1-beta)))
    term9_inner = -beta * (E_e/E_nu) * (gV**2 - gA**2) + ((gV**2 + gA**2)*(E_e/E_nu) - (me**2 + Q2)/(4*E_nu**2) * (gV**2 - gA**2)) * np.log(np.abs(log_arg))
    term9 = 2 * (mp/pi) * term9_factor * term9_inner
   
    return 1e16 * hc**2 * alpha/pi * GF**2 * Vud**2 * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9)

def result2D(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, f, hc, GF, Vud, alpha):
    """
    2-dimensional differential cross section in positron energy and angle for radiative IBD.
    Parameters
    ----------
    mp, mn, me : float
        Masses of proton, neutron, and positron (GeV).
    gV, gA : float
        Vector and axial-vector coupling constants.
    rV2, rA2 : float
        Squared isovector-vector and axial-vector radii (fm²).
    muV : float
        Isovector-vector magnetic moment.
    E_nu : float
        Antineutrino energy (GeV).
    E_e : float
        Positron energy (GeV).
    f : float
        Variable for the positron scattering angle (GeV).
    hc : float
        Reduced Planck constant times speed of light (GeV·fm).
    GF : float
        Fermi coupling constant (GeV⁻²).
    Vud : float
        CKM matrix element.
    alpha : float
        Fine-structure constant.
    Returns
    -------
    float
        2D differential cross section (10⁻⁴² cm²/GeV²).
    """
    pi = np.pi

    l0 = mp + E_nu - E_e
    E0 = (mn**2 + me**2 - mp**2) / (2 * mn)
    Delta = mp * E_nu - mn * E0
    s = mp**2 + 2 * mp * E_nu
    Sigma = (s - (me - mn)**2) * (s - (me + mn)**2)
    Q2 = mp**2 - mn**2 - 2 * mp * (E_e - E_nu)
    beta = np.sqrt(1 - me**2 / E_e**2)

    cosDelta = (E_nu**2 - (E_e**2 - me**2) - f**2) / (2 * f * np.sqrt(E_e**2 - me**2))
    l2 = l0**2 - f**2

    g = (f * cosDelta - beta * l0)**2 + (1 - beta**2) * f**2 * (1 - cosDelta**2)

    # Components split for clarity:
    term1 = -((f * beta) / g) * E_nu * (beta * l0 - f * cosDelta
            - f * (f - beta * l0 * cosDelta) / (2 * beta * E_e) * (1 - mn**2 / l2)) * (gV**2 + gA**2)

    term2 = (f/2
             - (f * beta) / (2 * g) * mp * (beta * l0 - f * cosDelta)
             - f/4 * (1 - mn**2 / l2)
             + f**2 / g * mp * (f - beta * l0 * cosDelta) / (4 * E_e) * (1 - mn**2 / l2)) * (gV**2 - gA**2)

    term3 = (-(E_nu/2) * (gV**2 + gA**2) + (E_nu - E_e)/4 * (gV**2 - gA**2)) * np.log(np.abs((l0 + f)/(l0 - f)))

    term4 = -f / (4 * beta) * (gV**2 - gA**2) * np.log(np.abs((1 + beta)/(1 - beta)))

    sqrt_g = np.sqrt(g)
    log_arg = (l0 - beta * f * cosDelta - sqrt_g) / (l0 - beta * f * cosDelta + sqrt_g)

    term5_num = (E_nu * (4 * g * E_e**2 + beta * E_e * (beta * l0 - f * cosDelta) * (l2 - mn**2)
                - 2 * f * me**2 * (f - beta * cosDelta * l0)) * (gV**2 + gA**2)
                + ((mp / (2 * E_nu)) * E_nu * (4 * g * E_e**2 + beta * E_e * (beta * l0 - f * cosDelta) * (l2 - mn**2)
                - 2 * f * me**2 * (f - beta * cosDelta * l0)) + (mn * E0 - mp * E_nu) * E_e * g) * (gV**2 - gA**2))

    term5 = -f / (4 * g**(3/2) * E_e**2) * np.log(np.abs(log_arg)) * term5_num

    term6 = -2 * (1 - 1/(2*beta) * np.log(np.abs((1 + beta)/(1 - beta)))) * (2*f) / (l2 - mn**2) * (
                E_nu * E_e * (gV**2 + gA**2) + ((E_e - E_nu)**2 - f**2 - me**2) / 4 * (gV**2 - gA**2)
            )

    prefactor = 1e16 * hc**2 * alpha / pi * GF**2 * Vud**2 * mp / (pi * E_nu**2)

    return prefactor * (term1 + term2 + term3 + term4 + term5 + term6)

def main():
    """
    Demonstration and test routine for inverse beta decay (IBD) spectra and radiative corrections.

    This function retrieves fundamental physical constants (such as particle and lepton masses, coupling constants,
    radii, magnetic moments, and other parameters) from scipy.constants, sets up example kinematic variables
    (neutrino and positron energies, angular variables, soft-photon cutoff), and computes a variety of spectra for
    IBD using the provided theoretical functions.

    Specifically, it performs the following tasks:
    - Sets up all necessary physical constants for IBD calculations, including:
        * mp, mn, me: Masses of proton, neutron, and positron (GeV)
        * GF: Fermi coupling constant (GeV⁻²)
        * hc: Reduced Planck constant times speed of light (GeV·fm)
        * alpha: Fine-structure constant
        * muV: Isovector-vector magnetic moment
        * Vud: CKM matrix element
        * gV, gA: Vector and axial-vector coupling constants
        * rV2, rA2: Squared isovector-vector and axial-vector radii (fm²)
    - Sets up kinematic variables for antineutrino and positron energies, angular variable, and soft-photon energy cutoff.
    - Computes and prints:
        * The 2D (positron energy and angle) differential cross section in radiative IBD (via result2D)
        * The inelastic positron energy spectrum (via result_inelastic)
        * The QED radiative correction to elastic positron energy spectrum (via result_elastic)
        * The QED radiative correction to electromagnetic energy spectrum with phase-space integration in the static limit (via result_EM_static)
        * The QED radiative correction to electromagnetic energy spectrum (via result_EM)
        * The leading-order positron energy spectrum with recoil, weak magnetism, and nucleon structure corrections (via result_LO)

    The function is intended as a demonstration: it prints out the computed spectra for the chosen example kinematics.
    The output allows users to verify the correctness and behavior of the theoretical routines provided in the script.

    Note:
        - The function expects all the required calculation functions (result2D, result_inelastic, result_elastic,
          result_EM_static, result_EM, result_LO, delta_v, f2, delta_s, delta_2p, delta_2_em, delta_1, delta_2) to be defined in the same module.
        - The function does not return any value; it outputs results to standard output.

    Example:
        When run as a script, this function prints the computed cross sections for typical IBD kinematics
        and serves as a test of the physical routines in the module.
    """
    mp = constants.physical_constants['proton mass energy equivalent in MeV'][0]/1000       # proton mass in GeV
    mn = constants.physical_constants['neutron mass energy equivalent in MeV'][0]/1000      # neutron mass in GeV
    me = constants.physical_constants['electron mass energy equivalent in MeV'][0]/1000     # positron mass in GeV
    GF = constants.physical_constants['Fermi coupling constant'][0]                         # Fermi coupling constant in GeV⁻²
    hc = constants.physical_constants['reduced Planck constant times c in MeV fm'][0]/1000  # GeV·fm (Reduced Planck constant times speed of light)
    alpha = 1./(1./constants.physical_constants['fine-structure constant'][0]+1./(3.*np.pi))# fine-structure constant
    muV = constants.physical_constants['proton mag. mom. to nuclear magneton ratio'][0] - constants.physical_constants['neutron mag. mom. to nuclear magneton ratio'][0]                                                                          # isovector-vector magnetic moment
    Vud = 0.97348                                                                           # CKM matrix element
    gV = 1.02499                                                                            # gV coupling constant
    lambdaA = 1.27660									    # ratio of gA/gV coupling constants
    gA = lambdaA * gV                                                                       # gA coupling constant
    rV2 = 0.578                                                                             # squared isovector-vector radius in fm²
    rA2 = 0.48                                                                              # squared axial-vector radius in fm²
    E_nu = 0.005                                                                            # antineutrino energy in GeV
    E_e = 0.002                                                                             # positron energy in GeV
    f = 0.0035                                                                              # variable f in GeV
    epsilon = 1e-10                                                                         # soft-photon energy cutoff in GeV

    result = result2D(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, f, hc, GF, Vud, alpha)
    print(f"2D distribution: {result}")
    result = result_inelastic(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, hc, GF, Vud, alpha)
    print(f"Positron energy spectrum, inelastic kinematics: {result}")
    E_e = 0.00368                                                                           # positron energy in GeV
    result = result_elastic(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, epsilon, hc, GF, Vud, alpha, delta_v, f2, delta_s, delta_2p)
    print(f"Radiative correction to positron energy spectrum, elastic kinematics: {result}")
    result = result_EM_static(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, epsilon, hc, GF, Vud, alpha, delta_1, delta_2)
    print(f"Radiative correction to electromagnetic energy spectrum with phase-space integration in the static limit: {result}")
    result = result_EM(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, epsilon, hc, GF, Vud, alpha, delta_v, f2, delta_s, delta_2_em)
    print(f"Radiative correction to electromagnetic energy spectrum: {result}")
    result = result_LO(mp, mn, me, gV, gA, rV2, rA2, muV, E_nu, E_e, hc, GF, Vud)
    print(f"Leading-order result with recoil, weak magnetism, and nucleon structure corrections: {result}")

if __name__ == "__main__":
    main()
