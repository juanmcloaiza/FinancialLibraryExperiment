import numpy as np
import scipy.optimize
import scipy.special


def discount_factor(t_now, t_future, risk_free_rate):
    r = risk_free_rate
    t = t_now
    T = t_future
    return np.exp(-r * (T - t))


def __present_value_coupon_bond__(ytm, principal, maturity, present_time,
                                  coupons=None, coupon_dates=None):
    P = principal
    Ci = coupons
    T = maturity
    t = present_time
    ti = coupon_dates
    y = ytm

    principal_value = P * np.exp(- y * (T - t))

    if coupons is None and coupon_dates is None:
        return principal_value

    else:

        if t > min(ti):
            raise Exception("Present time ahead of coupon dates.")

        if len(Ci) != len(ti):
            raise Exception("lenghts of coupons and coupon dates differ.")

        coupons_value = sum(
            [Ci[i] * np.exp(-y * (ti[i] - t)) for i in range(len(Ci))])

        return principal_value + coupons_value


def yield_to_maturity(present_value, principal, maturity, present_time,
                      coupons=None, coupon_dates=None):
    """
    Present value should be 0 < Z < P = principal
    Present time should be t < T = maturity
    """
    P = principal
    T = maturity
    Z = present_value
    t = present_time
    Ci = coupons
    ti = coupon_dates

    yp = - np.log(Z / P) / (T - t)

    if coupons is None and coupon_dates is None:
        return yp

    else:

        if t > min(ti):
            raise Exception("Present time ahead of coupon dates.")

        if len(Ci) != len(ti):
            raise Exception("lenghts of coupons and coupon dates differ.")

        fun = lambda y: __present_value_coupon_bond__(y, P, T, t, Ci, ti) - Z

        ytm = scipy.optimize.fsolve(fun, Z / P * 1e-4)

        return ytm


def duration(present_value, principal, maturity, present_time, coupons=None,
             coupon_dates=None):
    P = principal
    Ci = coupons
    T = maturity
    t = present_time
    ti = coupon_dates
    ytm = yield_to_maturity(present_value, principal, maturity, present_time,
                            coupons, coupon_dates)

    V_ = 1.0 / present_value
    dcoup = sum([Ci[i] * (t - ti[i]) * np.exp(-ytm * (ti[i] - t)) for i in
                 range(len(Ci))])
    dp = P * (t - T) * np.exp(-y * (T - t))

    ndur = dcoup * V_ + dp * V_

    return - ndur


def convexity(present_value, principal, maturity, present_time, coupons=None,
              coupon_dates=None):
    P = principal
    Ci = coupons
    T = maturity
    t = present_time
    ti = coupon_dates
    ytm = yield_to_maturity(present_value, principal, maturity, present_time,
                            coupons, coupon_dates)

    V_ = 1.0 / present_value
    dcoup = sum(
        [Ci[i] * np.power(t - ti[i], 2) * np.exp(-ytm * (ti[i] - t)) for i in
         range(len(Ci))])
    dp = P * np.power(t - T, 2) * np.exp(-ytm * (T - t))

    convex = dcoup * V_ + dp * V_

    return convex


def bs_call(stock_price, time_now, strike, expiry, risk_free_rate, volatility,
            dividend):
    if np.any(np.array(time_now - expiry) > 0):
        raise ValueError("Expiry in the past!")

    S = stock_price
    E = strike
    t = time_now
    T = expiry
    r = risk_free_rate
    sigma = volatility
    D = dividend

    d1num = np.log(S / E) + (r - D + 0.5 * sigma ** 2) * (T - t)
    d1den = sigma * np.sqrt(T - t)
    d1 = d1num / d1den
    d2 = d1 - sigma * np.sqrt(T - t)

    NormalCum = lambda x: 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2)))

    return S * np.exp(-D * (T - t)) * NormalCum(d1) - \
           E * np.exp(-r * (T - t)) * NormalCum(d2)


def bs_put(stock_price, time_now, strike, expiry, risk_free_rate, volatility,
           dividend):
    if np.any(np.array(time_now - expiry) > 0):
        raise ValueError("Expiry in the past!")

    S = stock_price
    E = strike
    t = time_now
    T = expiry
    r = risk_free_rate
    sigma = volatility
    D = dividend

    d1num = np.log(S / E) + (r - D + 0.5 * sigma ** 2) * (T - t)
    d1den = sigma * np.sqrt(T - t)
    d1 = d1num / d1den
    d2 = d1 - sigma * np.sqrt(T - t)

    NormalCum = lambda x: 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2)))

    return - S * np.exp(-D * (T - t)) * NormalCum(-d1) + E * np.exp(
        -r * (T - t)) * NormalCum(-d2)

def bs_straddle(stock_price, time_now, strike, expiry, risk_free_rate,
                volatility, dividend):
    put = bs_put(stock_price, time_now, strike, expiry, risk_free_rate,
                 volatility, dividend)
    call = bs_call(stock_price, time_now, strike, expiry, risk_free_rate,
                   volatility, dividend)
    return put + call


def bs_call_delta(stock_price, time_now, strike, expiry, risk_free_rate,
                volatility, dividend):
    if np.any(np.array(time_now - expiry) > 0):
        raise ValueError("Expiry in the past!")

    S = stock_price
    E = strike
    t = time_now
    T = expiry
    r = risk_free_rate
    sigma = volatility
    D = dividend

    d1num = np.log(S / E) + (r - D + 0.5 * sigma ** 2) * (T - t)
    d1den = sigma * np.sqrt(T - t)
    d1 = d1num / d1den
    d2 = d1 - sigma * np.sqrt(T - t)

    NormalCum = lambda x: 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2)))

    return np.exp(-D * (T - t)) * NormalCum(d1)


def bs_put_delta(stock_price, time_now, strike, expiry, risk_free_rate,
               volatility, dividend):
    if np.any(np.array(time_now - expiry) > 0):
        raise ValueError("Expiry in the past!")

    S = stock_price
    E = strike
    t = time_now
    T = expiry
    r = risk_free_rate
    sigma = volatility
    D = dividend

    d1num = np.log(S / E) + (r - D + 0.5 * sigma ** 2) * (T - t)
    d1den = sigma * np.sqrt(T - t)
    d1 = d1num / d1den
    d2 = d1 - sigma * np.sqrt(T - t)

    NormalCum = lambda x: 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2)))

    return np.exp(-D * (T - t)) * (NormalCum(d1) - 1.0)
