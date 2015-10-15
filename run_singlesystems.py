from __future__ import division, print_function
import numpy as np
# import matplotlib.pyplot as plt

import singlesystems
reload(singlesystems)

if __name__ == '__main__':

    stlr = singlesystems.get_catalog("q1_q16_stellar")
    kois = singlesystems.get_catalog("q1_q16_koi")

    ss = singlesystems.Singletypes(stlr, kois)

    # comp = np.load('comp.npy')

    planetperiod = [50, 300]
    planetradius = [0.75, 2.5]
    # startemp = [4200, 6100]
    starradius = [0, 1.15]
    starmass = [-np.inf, np.inf]
    dataspan = [2.*365.25, np.inf]
    dutycycle = [0.6, np.inf]
    rrmscdpp07p5 = [0, 1000]
    requirestarmass = True

    # ss.samplesetup(
    #     planetperiod=planetperiod,
    #     planetradius=planetradius,
    #     startemp=startemp,
    #     starradius=starradius,
    #     starmass=starmass,
    #     dataspan=dataspan,
    #     dutycycle=dutycycle,
    #     rrmscdpp07p5=rrmscdpp07p5,
    #     requirestarmass=requirestarmass,
    #     comp=None
    #     )

    ss.samplesetup(
        planetperiod=planetperiod,
        planetradius=planetradius,
        startype='G',
        starradius=starradius,
        starmass=starmass,
        dataspan=dataspan,
        dutycycle=dutycycle,
        rrmscdpp07p5=rrmscdpp07p5,
        requirestarmass=requirestarmass,
        comp=None
        )

    theta_opt = ss.optimize()

    mc = ss.mcmc()

    occ = ss.return_occurrence_samples(mc, 1000,
                                       [0.5, 2.0], [50, 300])
