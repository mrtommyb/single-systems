from __future__ import division, print_function
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

try:
    from io import BytesIO 
except ImportError:
    from cStringIO import StringIO as BytesIO

from scipy.stats import gamma
from scipy.optimize import minimize

import emcee

from scipy.integrate import nquad


class Singlesystems(object):

    def __init__(self, stlr, kois):
        """
        stlr is the stellar catalog
        kois is the koi catalog
        """
        self.stlr = stlr
        self.kois = kois

    def samplesetup(self,
                    planetperiod=[-np.inf, np.inf],
                    planetradius=[-np.inf, np.inf],
                    startemp=[-np.inf, np.inf],
                    starradius=[-np.inf, np.inf],
                    starmass=[-np.inf, np.inf],
                    dataspan=[-np.inf, np.inf],
                    dutycycle=[-np.inf, np.inf],
                    rrmscdpp07p5=[-np.inf, np.inf],
                    requirestarmass=True,
                    periodgridspacing=57,
                    radiusgridspacing=61,
                    bounds=[(-5, 5), (-5, 5), (-5, 5)],
                    comp=None
                    ):
        """
        this will change the state of self.stlr and self.kois
        """

        self.planetperiod = planetperiod
        self.planetradius = planetradius
        self.bounds = bounds

        # make cuts on the stellar catalog
        m = (startemp[0] <= self.stlr.teff) & (self.stlr.teff <= startemp[1])
        m &= (starradius[0] <= self.stlr.radius) & (self.stlr.radius <= starradius[1])
        m &= (starmass[0] <= self.stlr.mass) & (self.stlr.mass <= starmass[1])

        m &= (dataspan[0] <= self.stlr.dataspan) & (self.stlr.dataspan <= dataspan[1])
        m &= (dutycycle[0] <= self.stlr.dutycycle) & (self.stlr.dutycycle <= dutycycle[1])
        m &= (rrmscdpp07p5[0] <= self.stlr.rrmscdpp07p5) & (self.stlr.rrmscdpp07p5 <= rrmscdpp07p5[1])

        if requirestarmass:
            m &= np.isfinite(self.stlr.mass)

        self.stlr = pd.DataFrame(self.stlr[m])
        self.selectedstars = len(self.stlr)

        # Join on the stellar list.
        self.kois = pd.merge(self.kois, self.stlr[["kepid"]],
                             on="kepid", how="inner")

        # make cuts based on the planets catalog
        m = self.kois.koi_pdisposition == "CANDIDATE"
        m &= (planetperiod[0] <= self.kois.koi_period) & (self.kois.koi_period <= planetperiod[1])
        m &= np.isfinite(self.kois.koi_prad) & (planetradius[0] <= self.kois.koi_prad) & (self.kois.koi_prad <= planetradius[1])

        self.kois = pd.DataFrame(self.kois[m])

        self.selectedkois = len(self.kois)

        self._setcompleteness(periodgridspacing, radiusgridspacing, comp)


    def _setcompleteness(self, periodgridspacing, radiusgridspacing, comp):
        self.cdpp_cols = [k for k in self.stlr.keys() if k.startswith("rrmscdpp")]
        self.cdpp_vals = np.array([k[-4:].replace("p", ".") for k in self.cdpp_cols], dtype=float)

        # Pre-compute and freeze the gamma function from Equation (5) in
        # Burke et al.
        self.pgam = gamma(4.65, loc=0., scale=0.98)
        self.mesthres_cols = [k for k in self.stlr.keys() if k.startswith("mesthres")]
        self.mesthres_vals = np.array([k[-4:].replace("p", ".") for k in self.mesthres_cols],
                                 dtype=float)

        period = np.linspace(self.planetperiod[0], self.planetperiod[1], periodgridspacing)
        rp = np.linspace(self.planetradius[0], self.planetradius[1], radiusgridspacing)
        self.period_grid, self.rp_grid = np.meshgrid(period, rp, indexing="ij")

        self.koi_periods = np.array(self.kois.koi_period)
        self.koi_rps = np.array(self.kois.koi_prad)
        self.vol = np.diff(self.period_grid, axis=0)[:, :-1] * np.diff(self.rp_grid, axis=1)[:-1, :]

        if comp is None:
            comp = np.zeros_like(self.period_grid)

            for _, star in self.stlr.iterrows():
                comp += self.get_completeness(star, self.period_grid, self.rp_grid, 0.0, with_geom=True)

            self.comp = comp

        else:
            self.comp = comp


    def optimize(self):
        theta_0 = np.array([np.log(0.75), -0.53218, -1.5])
        r = minimize(self.nll, theta_0, method="L-BFGS-B", bounds=self.bounds)
        return r.x

    def mcmc(self):
        
        theta_opt = self.optimize()

        ndim, nwalkers = len(theta_opt), 16

        pos = [theta_opt + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)

        # Burn in.
        pos, _, _ = sampler.run_mcmc(pos, 1000)
        sampler.reset()

        # Production.
        pos, _, _ = sampler.run_mcmc(pos, 4000)
        return sampler.flatchain

    def get_pdet(self, star, aor, period, rp, e):
        """
        Equation (5) from Burke et al. Estimate the detection efficiency
        for a transit.
        
        :param star:   a pandas row giving the stellar properties
        :param aor:    the dimensionless semi-major axis (scaled
                       by the stellar radius)
        :param period: the period in days
        :param rp:     the planet radius in Earth radii
        :param e:      the orbital eccentricity
        
        """
        tau = self.get_duration(period, aor, e) * 24.
        mes = self.get_mes(star, period, rp, tau)
        mest = np.interp(tau, self.mesthres_vals,
                         np.array(star[self.mesthres_cols],
                                  dtype=float))
        x = mes - 4.1 - (mest - 7.1)
        return self.pgam.cdf(x)

    def get_pwin(self, star, period):
        """
        Equation (6) from Burke et al. Estimates the window function
        using a binomial distribution.
        
        :param star:   a pandas row giving the stellar properties
        :param period: the period in days
        
        """
        M = star.dataspan / period
        f = star.dutycycle
        omf = 1.0 - f
        pw = 1 - omf**M - M*f*omf**(M-1) - 0.5*M*(M-1)*f*f*omf**(M-2)
        msk = (pw >= 0.0) * (M >= 2.0)
        return pw * msk

    def get_pgeom(self, aor, e):
        """
        The geometric transit probability.
        
        See e.g. Kipping (2014) for the eccentricity factor
        http://arxiv.org/abs/1408.1393
        
        :param aor: the dimensionless semi-major axis (scaled
                    by the stellar radius)
        :param e:   the orbital eccentricity

        """
        return 1. / (aor * (1 - e*e)) * (aor > 1.0)

    def get_completeness(self, star, period, rp, e, with_geom=True):
        """
        A helper function to combine all the completeness effects.
        
        :param star:      a pandas row giving the stellar properties
        :param period:    the period in days
        :param rp:        the planet radius in Earth radii
        :param e:         the orbital eccentricity
        :param with_geom: include the geometric transit probability?
        
        """
        aor = self.get_a(period, star.mass) / star.radius
        pdet = self.get_pdet(star, aor, period, rp, e)
        pwin = self.get_pwin(star, period)
        if not with_geom:
            return pdet * pwin
        pgeom = self.get_pgeom(aor, e)
        return pdet * pwin * pgeom

    # A double power law model for the population.
    def population_model(self, theta, period, rp):
        lnf0, beta, alpha = theta
        v = np.exp(lnf0) * np.ones_like(period)
        for x, rng, n in zip((period, rp),
                             (self.planetperiod, self.planetradius),
                             (beta, alpha)):
            n1 = n + 1
            v *= x**n*n1 / (rng[1]**n1-rng[0]**n1)
        return v


    # The ln-likelihood function given at the top of this post.

    def lnlike(self, theta):
        pop = self.population_model(theta, self.period_grid, self.rp_grid) * self.comp
        pop = 0.5 * (pop[:-1, :-1] + pop[1:, 1:])
        norm = np.sum(pop * self.vol)
        ll = np.sum(np.log(self.population_model(theta, self.koi_periods, self.koi_rps))) - norm
        return ll if np.isfinite(ll) else -np.inf

    # The ln-probability function is just propotional to the ln-likelihood
    # since we're assuming uniform priors.
    
    def lnprob(self, theta):
        # Broad uniform priors.
        for t, rng in zip(theta, self.bounds):
            if not rng[0] < t < rng[1]:
                return -np.inf
        return self.lnlike(theta)

    # The negative ln-likelihood is useful for optimization.
    # Optimizers want to *minimize* your function.
    def nll(self, theta):
        ll = self.lnlike(theta)
        return -ll if np.isfinite(ll) else 1e15




    def get_duration(self, period, aor, e):
        """
        Equation (1) from Burke et al. This estimates the transit
        duration in the same units as the input period. There is a
        typo in the paper (24/4 = 6 != 4).
        
        :param period: the period in any units of your choosing
        :param aor:    the dimensionless semi-major axis (scaled
                       by the stellar radius)
        :param e:      the eccentricity of the orbit
        
        """
        return 0.25 * period * np.sqrt(1 - e**2) / aor

    def get_a(self, period, mstar, Go4pi=2945.4625385377644/(4*np.pi*np.pi)):
        """
        Compute the semi-major axis of an orbit in Solar radii.
        
        :param period: the period in days
        :param mstar:  the stellar mass in Solar masses
        
        """
        return (Go4pi*period*period*mstar) ** (1./3)

    def get_delta(self, k, c=1.0874, s=1.0187):
        """
        Estimate the approximate expected transit depth as a function
        of radius ratio. There might be a typo here. In the paper it
        uses c + s*k but in the public code, it is c - s*k:
        https://github.com/christopherburke/KeplerPORTs
        
        :param k: the dimensionless radius ratio between the planet and
                  the star
        
        """
        delta_max = k*k * (c + s*k)
        return 0.84 * delta_max

    def get_mes(self,star, period, rp, tau, re=0.009171):
        """
        Estimate the multiple event statistic value for a transit.
        
        :param star:   a pandas row giving the stellar properties
        :param period: the period in days
        :param rp:     the planet radius in Earth radii
        :param tau:    the transit duration in hours
        
        """
        # Interpolate to the correct CDPP for the duration.
        cdpp = np.array(star[self.cdpp_cols], dtype=float)
        sigma = np.interp(tau, self.cdpp_vals, cdpp)

        # Compute the radius ratio and estimate the S/N.
        k = rp * re / star.radius
        snr = self.get_delta(k) * 1e6 / sigma
        
        # Scale by the estimated number of transits.
        ntrn = star.dataspan * star.dutycycle / period 
        return snr * np.sqrt(ntrn)

    # A double power law model for the population.
    def population_model2(self,period, rp, theta):
        v = self.population_model(theta, period, rp, )
        return v

    def return_occurrence(self, parameters, 
        planetradius, planetperiod):
        occ = nquad(self.population_model2, 
            [[planetperiod[0],planetperiod[1]], [planetradius[0], planetradius[1]]], args=[parameters])[0]
        return occ

    def return_occurrence_samples(self, samplechain, sampsize, 
        planetradius, planetperiod):
        samp = np.zeros(sampsize)
        for i,x in enumerate(
                    np.random.choice(range(len(samplechain)), 
                    size=sampsize)):
            samp[i] = nquad(self.population_model2, 
                [[planetperiod[0],planetperiod[1]], [planetradius[0], planetradius[1]]], args=[samplechain[x]])[0]

        self.samp = samp
        self.occurence_rate_median = np.median(samp)
        return samp

    def return_occurrence_sample_semi(self, samplechain, sampsize,
                                      planetradius, planetsemi, starmass):
        """
        planet semi is a pair of values with the inner and outer range
        in AU
        starmass is an point estimate right now in solar masses
        """
        G = 6.67408E-11
        AU = 1.496E+11
        planetsemi = np.array(planetsemi)
        planetsemi *= AU
        starmass *= 1.989E30
        planetperiod = ((4. * np.pi**2 * planetsemi**3) / (G * starmass))**0.5 / 86400.

        samp = np.zeros(sampsize)
        for i,x in enumerate(
                    np.random.choice(range(len(samplechain)), 
                    size=sampsize)):
            samp[i] = nquad(self.population_model2, 
                [[planetperiod[0],planetperiod[1]], [planetradius[0], planetradius[1]]], args=[samplechain[x]])[0]

        self.samp = samp
        return samp


class Singletypes(Singlesystems):

    def __init__(self, stlr, kois):
        """
        stlr is the stellar catalog
        kois is the koi catalog
        """
        self.stlr = stlr
        self.kois = kois
        super(Singlesystems, self).__init__()

    def samplesetup(self,
                    planetperiod=[-np.inf, np.inf],
                    planetradius=[-np.inf, np.inf],
                    startype='G',
                    starradius=[-np.inf, np.inf],
                    starmass=[-np.inf, np.inf],
                    dataspan=[-np.inf, np.inf],
                    dutycycle=[-np.inf, np.inf],
                    rrmscdpp07p5=[-np.inf, np.inf],
                    requirestarmass=True,
                    periodgridspacing=57,
                    radiusgridspacing=61,
                    bounds=[(-5, 5), (-5, 5), (-5, 5)],
                    comp=None
                    ):
        self.planetperiod = planetperiod
        self.planetradius = planetradius
        self.bounds = bounds

        # make cuts on the stellar catalog
        m = self.maskstartype(startype)
        m &= (starradius[0] <= self.stlr.radius) & (self.stlr.radius <= starradius[1])
        m &= (starmass[0] <= self.stlr.mass) & (self.stlr.mass <= starmass[1])

        m &= (dataspan[0] <= self.stlr.dataspan) & (self.stlr.dataspan <= dataspan[1])
        m &= (dutycycle[0] <= self.stlr.dutycycle) & (self.stlr.dutycycle <= dutycycle[1])
        m &= (rrmscdpp07p5[0] <= self.stlr.rrmscdpp07p5) & (self.stlr.rrmscdpp07p5 <= rrmscdpp07p5[1])

        if requirestarmass:
            m &= np.isfinite(self.stlr.mass)

        self.stlr = pd.DataFrame(self.stlr[m])
        self.selectedstars = len(self.stlr)

        # Join on the stellar list.
        self.kois = pd.merge(self.kois, self.stlr[["kepid"]],
                             on="kepid", how="inner")

        # make cuts based on the planets catalog
        m = self.kois.koi_pdisposition == "CANDIDATE"
        m &= (planetperiod[0] <= self.kois.koi_period) & (self.kois.koi_period <= planetperiod[1])
        m &= np.isfinite(self.kois.koi_prad) & (planetradius[0] <= self.kois.koi_prad) & (self.kois.koi_prad <= planetradius[1])

        self.kois = pd.DataFrame(self.kois[m])

        self.selectedkois = len(self.kois)

        self._setcompleteness(periodgridspacing, radiusgridspacing, comp)

    def maskstartype(self, startype):
        if startype == 'M':
            m = (0 <= self.stlr.teff) & (self.stlr.teff <= 3900)

        elif startype == 'K':
            m = (3900 < self.stlr.teff) & (self.stlr.teff <= 5300)

        elif startype == 'G':
            m = (5300 < self.stlr.teff) & (self.stlr.teff <= 6000)

        elif startype == 'F':
            m = (6000 < self.stlr.teff) & (self.stlr.teff <= 7500)

        elif startype == 'A':
            m = (7500 < self.stlr.teff) & (self.stlr.teff <= 10000)

        return m


















def get_catalog(name, basepath="data"):
    fn = os.path.join(basepath, "{0}.h5".format(name))
    if os.path.exists(fn):
        return pd.read_hdf(fn, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    print("Downloading {0}...".format(name))
    url = ("http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
           "nph-nstedAPI?table={0}&select=*").format(name)
    r = requests.get(url)
    if r.status_code != requests.codes.ok:
        r.raise_for_status()
    fh = BytesIO(r.content)
    df = pd.read_csv(fh)
    df.to_hdf(fn, name, format="t")
    return df