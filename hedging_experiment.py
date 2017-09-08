#!/usr/bin/env python

import VanillaPricing as vp
import numpy as np
import matplotlib.pyplot as plt


def run():
    s0 = 10.09  # AGL.MI, 2017-04-24
    drift = 0.0 # It is not negative
    strike = s0 # We buy the option at the money
    expiry = 31.0  # 30 days

    # http://www.global-rates.com/interest-rates/libor/european-euro/eur-libor-interest-rate-overnight.aspx
    # august 31 2017    -0.42686 %
    # august 30 2017    -0.42900 %
    # august 29 2017    -0.42900 %
    # august 25 2017    -0.42543 %
    # Libor? OIS? < 1% in any case, and the price of the option is not too sensitive.
    risk_free_rate = 0.00
    volatility = 0.0057  # day-based... 0.6% ?
    dividend = 0  # WTF should I put here?

    # Generate stock price time series, Log-Normal model:
    # r = (s2 - s1) / s1 = s2/s1 - 1
    # log(r + 1) = log(s2/s1) is normally distributed -- logrets
    # s[2] = s[1]*exp(logrets[2])
    logrets = np.random.normal(drift, volatility, 29)
    st = [s0]
    for i in range(len(logrets)):
        st += [st[-1] * np.exp(logrets[i])]
    stock = np.array(st)

    # Generate time series with:
    #  - The price of the ATM call option, bought at t = 0.
    #  - Delta of the option. 
    #  - The value of a daily rebalanced portfolio:
    #    A short position of the call + delta long positions of the stock. 
    time = np.linspace(0, expiry - 1, len(stock))
    callopt = []
    deltacall = []
    portfolio = []
    for s, ti in zip(stock, time):
        callopt += [vp.bs_call(s, ti, strike, expiry,
                               risk_free_rate, volatility, dividend)]
        deltacall += [vp.bs_call_delta(s, ti, strike, expiry,
                                       risk_free_rate, volatility, dividend)]
        portfolio += [ -callopt[-1] + deltacall[-1] * s]
    callopt   = np.array(callopt)
    deltacall = np.array(deltacall)
    portfolio = np.array(portfolio)


    return stock, callopt, deltacall, portfolio

if __name__ == '__main__':

    
    Nruns = 120
    gains = []
    for i in range(1, Nruns+1):
        stock, callopt, deltacall, portfolio = run();
                
        if True:
            print('=== RUN {:d} ==='.format(i))
            print('---  STOCK ---')
            print(stock)
            print('---  CALL  ---')
            print(callopt)
            print('---  DELTA ---')
            print(deltacall)
            print('---PORTFOLIO--')
            print(portfolio)
            print('--------------')
            fig = plt.figure()
            plt.plot(stock, 'k-')
            plt.plot(callopt, 'r-')
            plt.plot(portfolio, 'b-')
            plt.savefig("{:d}.png".format(i))
            plt.close(fig)
        
        gains += [(portfolio[-1] / portfolio[0]) - 1.0]
        
        
    gains = np.array(gains)

    print('=== GAINS ===')
    print(gains)
    print('=============')
    print("Sum gains = {:.2f}".format(gains.sum()))
    print("Average gain = {:.2f}".format(gains.mean()))
    print("Standard dev = {:.2f}".format(gains.std()))
    print("Minimum gain = {:.2f}".format(gains.min()))
    print("Maximum gain = {:.2f}".format(gains.max()))
    gg = np.sort(gains)
    print("Min: {:.2f}".format(gg[0]))
    print("Median: {:.2f}".format(gg[int(Nruns/2)]))
    print("Max: {:.2f}".format(gg[-1]))