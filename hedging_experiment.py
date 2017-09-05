#!/usr/bin/env python

import VanillaPricing as vp
import numpy as np
import matplotlib.pyplot as plt


def run():
    s0 = 1.432  # A2A.MI 2017-09-03
    drift = 0.0
    strike = s0
    expiry = 31.0  # 30 days

    # http://www.global-rates.com/interest-rates/libor/european-euro/eur-libor-interest-rate-overnight.aspx
    # august 31 2017 	-0.42686 %
    # august 30 2017 	-0.42900 %
    # august 29 2017 	-0.42900 %
    # august 25 2017 	-0.42543 %
    # Libor? OIS? < 1% in any case
    risk_free_rate = 0.00
    volatility = 0.16  # day-based... 1.5% ?
    dividend = 0  # WTF should I put here?

    # Generate stock price series:
    # r = (s2 - s1) / s1 = s2/s1 - 1
    # log(1 + r) = log(s2/s1) is normally distributed -- logrets
    # s2 = s1*exp(logrets[2])
    logrets = np.random.normal(drift, volatility, 29)
    st = [s0]
    for i in range(len(logrets)):
        st += [st[-1] * np.exp(logrets[i])]
    st = np.array(st)

    # Generate option prices using Black and Scholes:
    t = np.linspace(0, expiry - 1, len(st))
    callopt = []
    deltacall = []
    portfolio = []
    for s, ti in zip(st, t):
        callopt += [vp.bs_call(s, ti, strike, expiry,
                               risk_free_rate, volatility, dividend)]
        deltacall += [vp.bs_call_delta(s, ti, strike, expiry,
                                       risk_free_rate, volatility, dividend)]
        portfolio += [-callopt[-1] + deltacall[-1] * s]
    # print(str(ti) + ",\t" + str(s) + ",\t" + str(a[-1]))
    callopt = np.array(callopt)
    deltacall = np.array(deltacall)
    portfolio = np.array(portfolio)

    portfolio_0 = portfolio[0]
    portfolio_final = portfolio[-1]
    portfolio_gain = portfolio_final / portfolio_0 - 1.0

    if True:
        print('---  STOCK ---')
        print(st)
        print('---  CALL  ---')
        print(callopt)
        print('---  DELTA ---')
        print(deltacall)
        print('---PORTFOLIO--')
        print(portfolio)
        print('--------------')

        # plt.plot(t, st / st[0], 'k-')
        # plt.plot(t, callopt / callopt[0], 'r-')
        # plt.plot(t, portfolio / portfolio[0], 'b-')
        # plt.show()
    print("GAIN: ", str(portfolio_gain))
    return portfolio_gain

if __name__ == '__main__':

    gains = []
    for i in range(0, 100):
        gains += [run()]
    gains = np.array(gains)
    print(gains.mean())
    print(gains.std())
    print(gains.min())
    print(gains.max())
    gg = np.sort(gains)
    print(gg[0])
    print(gg[50])
    print(gg[99])
