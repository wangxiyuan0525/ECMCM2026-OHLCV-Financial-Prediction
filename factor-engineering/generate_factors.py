def generate_factors(dataset_name, data):
    """
    Generate final 250-dim factor set.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g. "dataset0").
    data : np.ndarray
        Shape (T, 5) OHLCV array: [Open, High, Low, Close, Volume].

    Returns
    -------
    factors : np.ndarray
        Shape (T, 250) float32 factor matrix.
    """
    import numpy as np
    import talib
    import bottleneck as bn
    from numpy.lib.stride_tricks import sliding_window_view

    # ============================================================
    # Utility functions
    # ============================================================
    def _lag(arr, n):
        res = np.empty_like(arr)
        res[:n] = np.nan
        res[n:] = arr[:-n]
        return res


    def _rolling_cov(X, Y, window):
        Yb = Y[:, None] if Y.ndim == 1 and X.ndim > 1 else Y
        mx = bn.move_mean(X, window=window, min_count=1, axis=0)
        my = bn.move_mean(Yb, window=window, min_count=1, axis=0)
        mxy = bn.move_mean(X * Yb, window=window, min_count=1, axis=0)
        return mxy - mx * my


    def _signed_power(x, p):
        return np.sign(x) * (np.abs(x) ** p)


    def _power_only(x, p):
        return np.maximum(x, 0) ** p


    def _ts_rank(x, window):
        T = len(x)
        result = np.full(T, 0.5, dtype=np.float32)
        if T <= window:
            return result
        sw = sliding_window_view(x.astype(np.float64), window)
        result[window - 1:] = (sw <= sw[:, -1:]).mean(axis=1)
        return result


    def _rolling_zscore(x, window):
        mean_w = bn.move_mean(x, window=window, min_count=1)
        std_w = bn.move_std(x, window=window, min_count=1)
        eps = 1e-8
        result = (x - mean_w) / (std_w + eps)
        return np.nan_to_num(result, nan=0.0).astype(np.float32)


    def _decay_linear_weights(window):
        weights = np.arange(window, 0, -1, dtype=np.float64)
        return weights / weights.sum()


    def _rolling_decay_linear(x, window):
        T = len(x)
        result = np.full(T, np.nan, dtype=np.float64)
        w = _decay_linear_weights(window)
        for t in range(window - 1, T):
            result[t] = np.dot(x[t - window + 1: t + 1], w)
        result[:window - 1] = bn.move_mean(x, window=window, min_count=1)[:window - 1]
        return np.nan_to_num(result, nan=0.0).astype(np.float32)


    def _rolling_beta(x, y, window):
        mean_x = bn.move_mean(x, window=window, min_count=1)
        mean_y = bn.move_mean(y, window=window, min_count=1)
        eps = 1e-8
        cov_xy = bn.move_mean(x * y, window=window, min_count=1) - mean_x * mean_y
        var_y = bn.move_var(y, window=window, min_count=1)
        return cov_xy / (var_y + eps)


    def _rolling_sum(arr, w):
        cs = np.cumsum(np.insert(np.nan_to_num(arr, 0.0), 0, 0.0))
        res = cs[w:] - cs[:-w]
        out = np.full(len(arr), np.nan, dtype=np.float32)
        out[w - 1:] = res
        return out


    def _rolling_window_private(arr, w):
        out = np.full((len(arr), w), np.nan, dtype=np.float32)
        out[w - 1:] = sliding_window_view(arr, w)
        return out


    # ============================================================
    # Core: Compute all base 191 factors
    # ============================================================
    def _compute_base_191(data):
        """Compute the full 191-dim base factor set. Returns (T, 191) float32."""
        eps = 1e-8
        W = 60
        T = data.shape[0]
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            O, H, L, C, V = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
            HL_range = H - L + eps
            C_lag1 = _lag(C, 1)
            V_lag1 = _lag(V, 1)
            Ret = np.zeros(T, dtype=np.float32)
            Ret[1:] = np.log(C[1:] / (C[:-1] + eps))
            dC = np.zeros(T, dtype=np.float32)
            dC[1:] = C[1:] - C[:-1]
            ret_win = _rolling_window_private(Ret, W)
            ret_std = np.std(ret_win, axis=1) + eps
            vwap_proxy = (O + H + L + C) / 4.0
            log_V_plus_1 = np.log(V + 1.0)
            f_base = np.full((T, 191), np.nan, dtype=np.float32)

            # Level 1: (0~21)
            abs_C_O = np.abs(C - O)
            f_base[:, 0] = abs_C_O / HL_range
            f_base[:, 1] = (H - np.maximum(O, C)) / HL_range
            f_base[:, 2] = (np.minimum(O, C) - L) / HL_range
            f_base[:, 3] = (O + C) / (H + L + eps) - 1.0
            f_base[:, 4] = (C - vwap_proxy) / (vwap_proxy + eps)
            f_base[:, 5] = Ret
            f_base[:, 6] = np.log((C + eps) / (_lag(C, 2) + eps))
            f_base[:, 7] = np.log((C + eps) / (_lag(C, 3) + eps))
            f_base[:, 8] = np.log((C + eps) / (_lag(C, 5) + eps))
            f_base[:, 9] = np.log((O + eps) / (C_lag1 + eps))
            f_base[:, 10] = Ret * _lag(Ret, 1)
            f_base[:, 11] = np.log((H + eps) / (L + eps))
            f_base[:, 12] = (O - C_lag1) / HL_range
            log_HL_sq = (np.log((H + eps) / (L + eps))) ** 2
            log_CO_sq = (np.log((np.maximum(C, O) + eps) / (np.minimum(C, O) + eps))) ** 2
            f_base[:, 13] = np.maximum(0.5 * log_HL_sq - (2.0 * np.log(2.0) - 1.0) * log_CO_sq, 0.0)
            f_base[:, 14] = log_V_plus_1
            f_base[:, 15] = np.log((V + 1.0) / (V_lag1 + 1.0))
            f_base[:, 16] = V * (abs_C_O / HL_range)
            f_base[:, 17] = np.abs(Ret) / (log_V_plus_1 + eps)
            f_base[:, 18] = ((2.0 * C - H - L) / HL_range) * log_V_plus_1
            f_base[:, 19] = (C - O) / (log_V_plus_1 + eps)
            f_base[:, 20] = abs_C_O / HL_range
            f_base[:, 21] = log_V_plus_1 / np.maximum(np.log((H + eps) / (L + eps)), eps)

            # Level 2: TA-Lib (22~165)
            C_64, H_64, L_64, O_64, V_64 = (C.astype(np.float64), H.astype(np.float64),
                                             L.astype(np.float64), O.astype(np.float64), V.astype(np.float64))
            tp_64 = (H_64 + L_64 + C_64) / 3.0
            bop = talib.BOP(O_64, H_64, L_64, C_64)
            obv = talib.OBV(C_64, V_64)
            tanh_slopes, hma_sgn_list = [], []
            col_ptr = 26
            for p in [5, 15, 30, 60]:
                atr = talib.ATR(H_64, L_64, C_64, timeperiod=p)
                natr = atr / (C_64 + eps)
                roc = talib.ROC(C_64, timeperiod=p)
                sma_v = talib.SMA(V_64, timeperiod=p)
                v_surge = V_64 / (sma_v + eps)
                kama = talib.KAMA(C_64, timeperiod=p)
                trix = talib.TRIX(C_64, timeperiod=p)
                rsi = talib.RSI(C_64, timeperiod=p)
                adx = talib.ADX(H_64, L_64, C_64, timeperiod=p)
                half_p, sqrt_p = max(2, p // 2), max(2, int(p ** 0.5))
                hma = talib.WMA(2 * talib.WMA(C_64, half_p) - talib.WMA(C_64, p), sqrt_p)
                hma_norm = (hma - C_64) / (C_64 + eps)
                kama_norm = (kama - C_64) / (C_64 + eps)
                ema_p = talib.EMA(C_64, timeperiod=p)
                ema_ret = (C_64 - ema_p) / (ema_p + eps)
                max_h, min_l = bn.move_max(H_64, p), bn.move_min(L_64, p)
                ichimoku = (((max_h + min_l) / 2.0) - C_64) / (C_64 + eps)
                vpats = ((hma - kama) / (kama + eps)) * np.exp(-10.0 * natr)
                aes = (hma - kama) / (atr + eps)
                vgta = (trix - _lag(trix.astype(np.float32), 1)) * np.log1p(v_surge)
                hma_sgn_list.append(np.sign(hma - _lag(hma.astype(np.float32), 1)))
                mom, cci = talib.MOM(C_64, p), talib.CCI(H_64, L_64, C_64, p)
                rsi_min, rsi_max = bn.move_min(rsi, p), bn.move_max(rsi, p)
                stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + eps)
                willr, cmo = talib.WILLR(H_64, L_64, C_64, p), talib.CMO(C_64, p)
                macd, _, _ = talib.MACD(C_64, fastperiod=half_p, slowperiod=p)
                ppo = talib.PPO(C_64, fastperiod=half_p, slowperiod=p)
                upper, middle, lower = talib.BBANDS(C_64, p, 2, 2)
                bb_width = (upper - lower) / (middle + eps)
                bb_pctb = (C_64 - lower) / (upper - lower + eps)
                vwap = talib.SUM(V_64 * tp_64, p) / (talib.SUM(V_64, p) + eps)
                vwap_prem = (C_64 - vwap) / (vwap + eps)
                mfi = talib.MFI(H_64, L_64, C_64, V_64, p)
                cmf = talib.SUM(V_64 * ((C_64 - L_64) - (H_64 - C_64)) / (H_64 - L_64 + eps), p) / (
                            talib.SUM(V_64, p) + eps)
                bop_ema = talib.EMA(bop, max(3, p // 3))
                ef = -roc * (bop - bop_ema)
                med_z = (ef - bn.move_mean(ef, p)) / (bn.move_std(ef, p) + eps)
                try:
                    import pandas as pd
                    kurt = pd.Series(Ret).rolling(p).kurt().values
                    skew = pd.Series(Ret).rolling(p).skew().values
                except ImportError:
                    kurt = np.full(T, np.nan, dtype=np.float32)
                    skew = np.full(T, np.nan, dtype=np.float32)
                capitulation = np.where((rsi < 30) & (adx > 35), 1.0, 0.0).astype(np.float32)
                tanh_slopes.append(
                    np.tanh(100.0 * (trix - _lag(trix.astype(np.float32), 1)) / (np.abs(_lag(trix.astype(np.float32), 1)) + eps)))
                f_base[:, col_ptr:col_ptr + 35] = np.column_stack([
                    hma_norm, kama_norm, ema_ret, trix, ichimoku, vpats, aes, vgta,
                    mom, roc, cci, rsi, stoch_rsi, willr, cmo, macd, ppo, roc / (natr * 100 + eps),
                    adx, talib.ADXR(H_64, L_64, C_64, p), talib.AROONOSC(H_64, L_64, p), natr, bb_width, bb_pctb,
                    vwap_prem, mfi, cmf, v_surge, bop_ema, med_z, med_z, med_z, kurt, skew, capitulation
                ])
                col_ptr += 35
            f_base[:, 22:26] = np.column_stack([np.sum(hma_sgn_list, 0), np.sum(tanh_slopes, 0), obv, bop])

            # Level 3: (166~190)
            def _calc_l3(win):
                s_vp = _rolling_sum(V * vwap_proxy, win)
                s_v = _rolling_sum(V, win)
                vw_w = s_vp / (s_v + eps)
                x_ou = C - vw_w
                z_ou = _lag(C - vw_w, 1)
                x_w = _rolling_window_private(x_ou, win)
                z_w = _rolling_window_private(z_ou, win)
                b_ou = (np.mean(x_w * z_w, 1) - np.mean(x_w, 1) * np.mean(z_w, 1)) / (np.var(z_w, 1) + eps)
                rv_w = np.sum(_rolling_window_private(Ret ** 2, win), 1)
                bpv_w = (np.pi / 2) * np.sum(_rolling_window_private(np.abs(Ret) * _lag(np.abs(Ret), 1), win), 1)
                zb = Ret / ret_std
                v_buy = V * (1.0 / (1.0 + np.exp(-1.702 * zb)))
                v_sell = V - v_buy
                svb = _rolling_sum(v_buy, win)
                svs = _rolling_sum(v_sell, win)
                vpin = np.abs(svb - svs) / (svb + svs + eps)
                nv = (v_buy - v_sell) / (s_v / win + eps)
                nv_w = _rolling_window_private(nv, win)
                dc_w = _rolling_window_private(dC, win)
                kl = (np.mean(nv_w * dc_w, 1) - np.mean(nv_w, 1) * np.mean(dc_w, 1)) / (np.var(nv_w, 1) + eps)
                return [1 - b_ou, rv_w, bpv_w, vpin, svb / (svb + svs + eps), kl]
            l3_60 = _calc_l3(60)
            l3_15 = _calc_l3(15)
            f_base[:, 166:172] = np.column_stack(l3_60)
            f_base[:, 172:178] = np.column_stack(l3_15)
            f_base[:, 178:] = np.nan_to_num(f_base[:, 166:179], 0.0)
        return f_base.astype(np.float32)


    # ============================================================
    # Group 1: SP Factors (70 dims, unchanged from v3)
    # ============================================================
    def _generate_sp_factors(base, C, H, L, O, V):
        """Generate all 70 SP factors. base = (T, 191)"""
        eps = 1e-8
        T = base.shape[0]
        sp_factors = []

        # --- SP15 (p=1.5) Round 1: 14 direction ---
        sp15_r1_targets = [
            base[:, 4],     # VWAP_Dev
            base[:, 18],    # Order_Imbalance_Px
            base[:, 5],     # LogRet
            base[:, 35],    # roc_5M
            base[:, 140],   # roc_60M
            base[:, 146],   # macd_60M
            base[:, 147],   # ppo_60M
            base[:, 134],   # trix_60M
            base[:, 136],   # vpats_60M
            base[:, 141],   # cci_60M
            base[:, 142] / 100.0 - 0.5,  # rsi_centered_60M
            base[:, 157],   # cmf_60M
            base[:, 0] - 0.5,      # Close_Position centered
            base[:, 1],     # Upper_Shadow_Frac
        ]
        for x in sp15_r1_targets:
            sp_factors.append(_signed_power(x, 1.5))

        # --- SP20 (p=2.0) Round 1: 9 (7 direction + 2 bounded) ---
        sp20_r1_targets = [
            base[:, 4], base[:, 5], base[:, 18], base[:, 35], base[:, 140],
            base[:, 146], base[:, 134],
        ]
        for x in sp20_r1_targets:
            sp_factors.append(_signed_power(x, 2.0))
        sp_factors.append(_power_only(base[:, 0], 2.0))
        body_frac = np.abs(base[:, 5]) / (np.abs(base[:, 11]) + eps)
        sp_factors.append(_power_only(np.clip(body_frac, 0, 1), 2.0))

        # --- SP05 (p=0.5) Round 1: 6 non-negative ---
        sp05_targets = [
            base[:, 13],  # GK_Vol
            base[:, 11],  # LogHL
            base[:, 152], # natr_60M
            base[:, 149], # adx_60M
            np.sqrt(bn.move_sum(base[:, 5]**2, window=5, min_count=1)),  # RV_5
            bn.move_mean(base[:, 11], window=60, min_count=1),  # LogHL_Roll60
        ]
        for x in sp05_targets:
            sp_factors.append(_power_only(x, 0.5))

        # --- SP03 (p=0.3) Round 1: 5 non-negative ---
        sp03_targets = [
            base[:, 13],  # GK_Vol
            base[:, 11],  # LogHL
            base[:, 152], # natr_60M
            base[:, 168], # Jump_Ratio
            base[:, 17],  # Amihud-like
        ]
        for x in sp03_targets:
            sp_factors.append(_power_only(np.abs(x) + eps, 0.3))

        # --- SP15 Round 2: 18 new signals ---
        C_64 = C.astype(np.float64)
        ema5 = talib.EMA(C_64, timeperiod=5)
        ema15 = talib.EMA(C_64, timeperiod=15)
        ema_ret_5 = (C_64 - ema5) / (ema5 + eps)
        ema_ret_15 = (C_64 - ema15) / (ema15 + eps)
        mom_5 = np.zeros(T, dtype=np.float32); mom_5[5:] = C[5:] - C[:-5]
        mom_60 = np.zeros(T, dtype=np.float32); mom_60[60:] = C[60:] - C[:-60]
        roc_15 = np.zeros(T, dtype=np.float32); roc_15[15:] = (C[15:] - C[:-15]) / (C[:-15] + eps)
        half, sqrt_p = 30, int(np.sqrt(60))
        wma_half = talib.WMA(C_64, timeperiod=half)
        wma_full = talib.WMA(C_64, timeperiod=60)
        hma_raw = 2 * wma_half - wma_full
        hma_dev = (talib.WMA(hma_raw, timeperiod=sqrt_p) - C_64) / (C_64 + eps)
        kama_dev = (talib.KAMA(C_64, timeperiod=60) - C_64) / (C_64 + eps)
        hh30, ll30 = bn.move_max(H, 30), bn.move_min(L, 30)
        ichimoku_prem = (C - (hh30 + ll30) / 2.0) / (((hh30 + ll30) / 2.0) + eps)
        willr = talib.WILLR(H.astype(np.float64), L.astype(np.float64), C_64, timeperiod=60)
        willr_norm = np.nan_to_num(willr, nan=-50) / 100.0
        cmo = talib.CMO(C_64, timeperiod=60) / 100.0
        bop_raw = np.where(H - L > eps, (C - O) / (H - L + eps), 0)
        bop_ema_60 = bn.move_mean(bop_raw, window=60, min_count=1)
        aroon_up = np.zeros(T, dtype=np.float32); aroon_down = np.zeros(T, dtype=np.float32)
        for t in range(60, T):
            wh, wl = H[t-60:t+1], L[t-60:t+1]
            aroon_up[t] = 100 * (60 - (60 - np.argmax(wh))) / 60
            aroon_down[t] = 100 * (60 - (60 - np.argmin(wl))) / 60
        aroon_osc = (aroon_up - aroon_down) / 100.0
        vol_roc = np.zeros(T, dtype=np.float32); vol_roc[20:] = (V[20:] - V[:-20]) / (V[:-20] + eps)
        dP = np.zeros(T, dtype=np.float32); dP[1:] = C[1:] - C[:-1]
        dV = np.zeros(T, dtype=np.float32); dV[1:] = V[1:] - V[:-1]
        pv_delta = (dP * dV) / (C * V + eps)
        roc5_r2 = np.zeros(T, dtype=np.float32); roc5_r2[5:] = (C[5:] - C[:-5]) / (C[:-5] + eps)
        roc60_r2 = np.zeros(T, dtype=np.float32); roc60_r2[60:] = (C[60:] - C[:-60]) / (C[:-60] + eps)
        roc_diff = roc5_r2 - roc60_r2
        log_r = np.zeros(T, dtype=np.float32); log_r[1:] = np.log(C[1:] / (C[:-1] + eps))
        rv5_r2 = bn.move_sum(log_r**2, window=5, min_count=1)
        rv60_r2 = bn.move_sum(log_r**2, window=60, min_count=1)
        rv_ratio = rv5_r2 / (rv60_r2 + eps) - 1.0
        bb_mid = bn.move_mean(C, window=60, min_count=1)
        bb_std = bn.move_std(C, window=60, min_count=1)
        bb_pos = np.where(bb_std > eps, (C - bb_mid) / (2 * bb_std + eps), 0)
        ohlc4 = (O + H + L + C) / 4.0
        vwap_num = bn.move_sum(V * ohlc4, window=60, min_count=1)
        vwap_den = bn.move_sum(V, window=60, min_count=1)
        vwap_spread = (C - vwap_num / (vwap_den + eps)) / (C + eps)

        sp15_r2_targets = [
            ema_ret_5, ema_ret_15, mom_5, mom_60, roc_15, hma_dev, kama_dev,
            ichimoku_prem, willr_norm, cmo, bop_ema_60, aroon_osc, vol_roc,
            pv_delta, roc_diff, rv_ratio, bb_pos, vwap_spread
        ]
        for x in sp15_r2_targets:
            sp_factors.append(_signed_power(x, 1.5))

        # --- SP20 Round 2: same 18 signals ---
        for x in sp15_r2_targets:
            sp_factors.append(_signed_power(x, 2.0))

        return np.column_stack(sp_factors).astype(np.float32)  # 70 factors


    # ============================================================
    # Group 2: wxy New Factors (22 dims, unchanged from v3)
    # ============================================================
    def _generate_wxy_new_factors(C, H, L, O, V):
        """Generate 22 wxy_new factors: 6 FracDiff + 3 KF + 1 POC + 12 Literature."""
        eps = 1e-8
        T = len(C)
        factors_list = []

        # --- Literature (12) ---
        hl_range = H - L
        parkinson = np.sqrt(np.log(H / np.maximum(L, eps))**2 / (4 * np.log(2)))
        factors_list.append(parkinson.reshape(-1, 1))
        rs_vol = np.sqrt(np.log(H / np.maximum(C, eps)) * np.log(H / np.maximum(O, eps)) +
                         np.log(L / np.maximum(C, eps)) * np.log(L / np.maximum(O, eps)))
        factors_list.append(np.nan_to_num(rs_vol, 0.0).reshape(-1, 1))
        lower_shadow = np.where(hl_range > eps, (np.minimum(O, C) - L) / hl_range, 0)
        factors_list.append(lower_shadow.reshape(-1, 1))
        body_frac = np.where(hl_range > eps, np.abs(C - O) / hl_range, 0)
        factors_list.append(body_frac.reshape(-1, 1))
        close_pos = np.where(hl_range > eps, (C - L) / hl_range, 0.5)
        factors_list.append(close_pos.reshape(-1, 1))
        dC = np.diff(C.astype(np.float64), prepend=C[:1].astype(np.float64))
        cov_dc = bn.move_mean(dC[1:] * dC[:-1], window=60, min_count=1)
        cov_dc_pad = np.pad(cov_dc, (1, 0), constant_values=0)
        roll_spread = np.where(cov_dc_pad < 0, 2 * np.sqrt(np.maximum(-cov_dc_pad, 0)), 0).astype(np.float32)
        factors_list.append(roll_spread.reshape(-1, 1))
        net_change = np.abs(C[60:] - C[:-60])
        path_length = bn.move_sum(np.abs(np.diff(C.astype(np.float64), prepend=C[:1].astype(np.float64))), window=60, min_count=1)[60:]
        kaufman_er = np.zeros(T, dtype=np.float32)
        kaufman_er[60:] = np.where(path_length > eps, net_change / (path_length + eps), 0)
        factors_list.append(kaufman_er.reshape(-1, 1))
        hh60, ll60 = bn.move_max(H, 60), bn.move_min(L, 60)
        donchian_w = np.where(C > eps, (hh60 - ll60) / C, 0)
        factors_list.append(donchian_w.reshape(-1, 1))
        dP_small = np.diff(C.astype(np.float64), prepend=C[:1].astype(np.float64))
        mean_dP = bn.move_mean(dP_small, window=60, min_count=1)
        mean_V = bn.move_mean(V, window=60, min_count=1)
        cov_PV = bn.move_mean((dP_small - mean_dP) * (V - mean_V), window=60, min_count=1)
        std_P = bn.move_std(dP_small, window=60, min_count=1)
        std_V = bn.move_std(V, window=60, min_count=1)
        pv_corr = np.where((std_P * std_V) > eps, cov_PV / (std_P * std_V + eps), 0)
        factors_list.append(pv_corr.reshape(-1, 1))
        vw_num = bn.move_sum(V * C, window=60, min_count=1)
        vw_den = bn.move_sum(V, window=60, min_count=1)
        vw_close = vw_num / (vw_den + eps)
        factors_list.append(vw_close.reshape(-1, 1))
        log_ret = np.zeros(T, dtype=np.float32); log_ret[1:] = np.log(C[1:] / (C[:-1] + eps))
        rv5 = np.sqrt(bn.move_sum(log_ret**2, window=5, min_count=1))
        factors_list.append(rv5.reshape(-1, 1))
        mom60 = np.zeros(T, dtype=np.float32); mom60[60:] = (C[60:] - C[:-60]) / (C[:-60] + eps)
        std60 = bn.move_std(log_ret, window=60, min_count=1)
        ram = np.where(std60 > eps, mom60 / (std60 + eps), 0)
        factors_list.append(ram.reshape(-1, 1))

        # --- FracDiff (6) ---
        def frac_diff_fast(series, d=0.4, p=60):
            result = np.zeros(T, dtype=np.float32)
            if T <= p: return result
            w = np.zeros(p, dtype=np.float64); w[0] = 1.0
            for k in range(1, p): w[k] = -w[k-1] * (d - k + 1) / k
            sw = sliding_window_view(series.astype(np.float64), p)
            result[p-1:] = np.dot(sw, w[::-1])
            return result
        log_C = np.log(np.maximum(C, eps))
        factors_list.append(frac_diff_fast(log_C).reshape(-1, 1))
        midline = (hh60 + ll60) / 2.0
        log_midline = np.log(np.maximum(midline, eps))
        factors_list.append(frac_diff_fast(log_midline).reshape(-1, 1))
        ema60 = bn.move_mean(C, window=60, min_count=1)
        log_ema = np.log(np.maximum(ema60, eps))
        factors_list.append(frac_diff_fast(log_ema).reshape(-1, 1))
        ohlc4 = (O + H + L + C) / 4.0
        vwap_num2 = bn.move_sum(V * ohlc4, window=60, min_count=1)
        vwap_den2 = bn.move_sum(V, window=60, min_count=1)
        vwap60 = vwap_num2 / (vwap_den2 + eps)
        vwap_prem = (C - vwap60) / (vwap60 + eps)
        factors_list.append(frac_diff_fast(vwap_prem).reshape(-1, 1))
        hma_s = bn.move_mean(C, window=30, min_count=1)
        kama_s = bn.move_mean(C, window=60, min_count=1)
        atr_s = bn.move_mean(np.abs(np.diff(C.astype(np.float64), prepend=C[:1].astype(np.float64))), window=60, min_count=1)
        natr_s = np.where(C > eps, atr_s / C, 0)
        vpats_s = np.where(np.abs(kama_s) > eps, ((hma_s - kama_s) / kama_s) * np.exp(-10 * natr_s), 0)
        factors_list.append(frac_diff_fast(vpats_s).reshape(-1, 1))
        rsi_c = np.nan_to_num(talib.RSI(C.astype(np.float64), timeperiod=60), nan=50)
        rsi_centered = (rsi_c - 50) / 50
        factors_list.append(frac_diff_fast(rsi_centered).reshape(-1, 1))

        # --- KF (3) ---
        def kf_1d(obs, Q=1e-4, R=1e-2):
            Tk = len(obs); x_hat = np.zeros(Tk, dtype=np.float64); P = np.zeros(Tk)
            x_hat[0] = obs[0]; P[0] = 1.0
            for t in range(1, Tk):
                x_pred = x_hat[t-1]; P_pred = P[t-1] + Q
                K = P_pred / (P_pred + R)
                x_hat[t] = x_pred + K * (obs[t] - x_pred)
                P[t] = (1 - K) * P_pred
            return x_hat.astype(np.float32), P.astype(np.float32)
        kf_state, _ = kf_1d(C)
        factors_list.append(((C - kf_state) / (kf_state + eps)).reshape(-1, 1))
        level = np.zeros(T, dtype=np.float64); trend = np.zeros(T)
        level[0] = C[0]; trend[0] = 0
        P00, P11 = np.ones(T), np.ones(T)
        for t in range(1, T):
            l_pred = level[t-1] + trend[t-1]; t_pred = trend[t-1]
            P00p = P00[t-1] + P11[t-1] + 1e-4; P11p = P11[t-1] + 1e-6
            y = C[t] - l_pred; S = P00p + 1e-2
            K0 = P00p / S
            level[t] = l_pred + K0 * y; trend[t] = t_pred
            P00[t] = (1 - K0) * P00p; P11[t] = P11p
        factors_list.append((trend / (C + eps)).astype(np.float32).reshape(-1, 1))
        spread = C - vwap60
        spread_state, spread_P = kf_1d(spread, Q=1e-5, R=1e-2)
        factors_list.append(((spread - spread_state) / (np.sqrt(spread_P) + eps)).reshape(-1, 1))

        # --- POC (1) ---
        poc_dist = (C - vwap60) / (atr_s + eps)
        factors_list.append(poc_dist.reshape(-1, 1))

        return np.nan_to_num(np.concatenate(factors_list, axis=1), 0.0).astype(np.float32)  # 22 factors


    # ============================================================
    # Group 3: A-H Factors PRUNED (15 dims, down from 25)
    # ============================================================
    def _generate_ah_factors_pruned(base, C, H, L, O, V):
        """Generate 15 pruned A-H factors (down from 25)."""
        eps = 1e-8
        T = len(C)
        gk_vol = base[:, 13]
        log_hl_range = base[:, 11]
        close_pos = base[:, 0]
        log_ret = base[:, 5]
        order_imb = base[:, 18]
        vwap_dev = base[:, 4]
        roc_5 = base[:, 35]
        roc_60 = base[:, 140]
        macd = base[:, 146]
        ema60_simple = bn.move_mean(C, window=60, min_count=1)
        ema_ret = (C - ema60_simple) / (ema60_simple + eps)

        ah = np.zeros((T, 25), dtype=np.float32)
        # A. TsRank (4) — all kept
        ah[:, 0] = _ts_rank(gk_vol, 5)
        ah[:, 1] = _ts_rank(order_imb, 20)
        ah[:, 2] = _ts_rank(log_hl_range, 60)
        ah[:, 3] = _ts_rank(vwap_dev, 10)
        # B. ZScore (4) — all kept
        ah[:, 4] = _rolling_zscore(gk_vol, 60)
        ah[:, 5] = _rolling_zscore(order_imb, 60)
        ah[:, 6] = _rolling_zscore(log_hl_range, 60)
        ah[:, 7] = _rolling_zscore(close_pos, 30)
        # C. VolScaled — keep 3 of 4
        gk_ma5 = bn.move_mean(gk_vol, window=5, min_count=1)
        gk_ma20 = bn.move_mean(gk_vol, window=20, min_count=1)
        gk_ma60 = bn.move_mean(gk_vol, window=60, min_count=1)
        ah[:, 8] = roc_5 / (np.sqrt(gk_ma5) + eps)
        ah[:, 9] = roc_60 / (np.sqrt(gk_ma60) + eps)
        ah[:, 10] = macd / (np.sqrt(gk_ma60) + eps)
        ah[:, 11] = order_imb / (np.sqrt(gk_ma20) + eps)
        # D. DecayLinear — keep 2 of 3
        ah[:, 12] = _rolling_decay_linear(log_ret, 5)
        ah[:, 13] = _rolling_decay_linear(gk_vol, 10)
        ah[:, 14] = _rolling_decay_linear(log_hl_range, 20)
        # E. Orth — keep 2 of 3
        beta1 = _rolling_beta(ema_ret, gk_vol, 60)
        ah[:, 15] = ema_ret - beta1 * gk_vol
        beta2 = _rolling_beta(vwap_dev, log_hl_range, 60)
        ah[:, 16] = vwap_dev - beta2 * log_hl_range
        beta3 = _rolling_beta(order_imb, gk_vol, 60)
        ah[:, 17] = order_imb - beta3 * gk_vol
        # F. Delta (3) — all dropped
        ah[3:, 18] = order_imb[3:] - order_imb[:-3]
        ah[5:, 19] = gk_vol[5:] - gk_vol[:-5]
        ah[5:, 20] = log_hl_range[5:] - log_hl_range[:-5]
        # G. SignedPower (2) — all dropped
        ah[:, 21] = _signed_power(log_ret, 2.0)
        ah[:, 22] = _signed_power(vwap_dev, 1.5)
        # H. Cross (2) — all dropped
        ah[:, 23] = order_imb * gk_vol
        ah[:, 24] = vwap_dev * log_hl_range

        keep_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16]
        return np.nan_to_num(ah[:, keep_idx], 0.0).astype(np.float32)


    # ============================================================
    # Group 4: L4 Derived PRUNED (4 dims, down from 20)
    # ============================================================
    def _generate_l4_derived_pruned(base):
        """Generate 4 pruned L4 derived factors (down from 20)."""
        eps = 1e-8; W = 60; T = base.shape[0]
        core = np.nan_to_num(base[:, 0:1], 0.0)
        anch_v = np.nan_to_num(base[:, 13:14], 0.0)
        anch_r5 = np.nan_to_num(base[:, 133:134], 0.0)

        # Op1: Residual (ClosePos)
        m_v = bn.move_mean(anch_v[:, 0], W, 1, 0)
        v_v = np.clip(bn.move_mean(anch_v[:, 0] ** 2, W, 1, 0) - m_v ** 2, eps, None)
        beta = _rolling_cov(core, anch_v[:, 0], W) / v_v[:, None]
        res = core - (bn.move_mean(core, W, 1, 0) - beta * m_v[:, None] + beta * anch_v)
        op1 = np.clip(np.nan_to_num(res, 0.0), -1e4, 1e4)

        # Op2: Phase angle (ClosePos)
        d3 = core - _lag(core, 3); d15 = core - _lag(core, 15)
        op2 = np.nan_to_num(np.arctan(d3 / (d15 + eps)), 0.0)

        # Op3: Z-Score (ClosePos)
        s_c = np.clip(bn.move_std(core, W, 1, 0), eps, None)
        zsc = (core - bn.move_mean(core, W, 1, 0)) / s_c
        op3 = np.clip(np.nan_to_num(zsc, 0.0), -5.0, 5.0)

        # Op4: Resonance (ClosePos)
        op4 = np.clip(np.nan_to_num(core * anch_r5, 0.0), -1e6, 1e6)

        return np.concatenate([op1, op2, op3, op4], axis=1).astype(np.float32)


    # ============================================================
    # Group 5: Top 120 Base Factors (extended from 100)
    # ============================================================
    _BASE_TOP_120_IDX = [
        8, 35, 7, 4, 6, 5, 133, 18, 19, 11, 28, 13,
        63, 70, 30, 105, 34, 69, 131, 139, 140,
        166, 167, 168, 171, 174, 175,
        0, 1, 2, 3, 9, 10, 12, 14, 15, 16, 17, 20, 21,
        22, 23, 24, 25, 26, 27, 29, 31, 32, 33,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
        64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 106, 107, 108, 109,
    ]


    # ============================================================
    # Group 6: Extra SP Variants (15 dims, SP15 only)
    # ============================================================
    _EXTRA_SP_BASE_IDX = [
        4, 5, 6, 7, 8, 18, 35, 133, 139, 140, 166, 167, 168, 174, 175,
    ]


    # ============================================================
    # Main logic
    # ============================================================
    eps = 1e-8
    O, H, L, C, V = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    base = _compute_base_191(data)

    blocks = []

    # Group 1: All 70 SP factors (core, proven ~50%+ contribution)
    blocks.append(_generate_sp_factors(base, C, H, L, O, V))  # 70

    # Group 2: All 22 wxy_new factors (FracDiff+KF+POC+Literature)
    blocks.append(_generate_wxy_new_factors(C, H, L, O, V))  # 22

    # Group 3: Pruned A-H factors (15, down from 25)
    blocks.append(_generate_ah_factors_pruned(base, C, H, L, O, V))  # 15

    # Group 4: Pruned L4 derived (4, down from 20)
    blocks.append(_generate_l4_derived_pruned(base))  # 4

    # Group 5: Top 120 base factors (extended from 100)
    blocks.append(base[:, _BASE_TOP_120_IDX])  # 120

    # Group 6: Extra SP variants (SP15 only on 15 best base)
    extra_sp = [_signed_power(base[:, i], 1.5) for i in _EXTRA_SP_BASE_IDX]
    blocks.append(np.column_stack(extra_sp))  # 15

    # Group 7: Cross-features (4 interactions of top SP factors)
    sp = blocks[0]  # (T, 70)
    cross = np.column_stack([
        sp[:, 4] * sp[:, 0],    # roc_60M^1.5 × VWAP_Dev^1.5
        sp[:, 4] * sp[:, 3],    # roc_60M^1.5 × roc_5M^1.5
        sp[:, 7] * sp[:, 25],   # trix_60M^1.5 × natr_60M^0.5
        sp[:, 38] * sp[:, 41],  # roc_15^1.5 × kama_dev^1.5 (R2)
    ])
    blocks.append(np.nan_to_num(cross, 0.0).astype(np.float32))  # 4

    factors = np.concatenate(blocks, axis=1)
    if factors.shape[1] != 250:
        raise RuntimeError(f"Expected 250 factors, got {factors.shape[1]}")
    return np.nan_to_num(factors, 0.0).astype(np.float32)
