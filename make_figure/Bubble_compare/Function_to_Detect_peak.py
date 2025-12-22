import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, fftconvolve
from astropy.stats import sigma_clipped_stats



def find_verified_peak(c18o_spec, co12_spec, co13_spec, vaxis, dv,
                       smoothing_sigma=1.0,
                       c18o_height_factor=5.0, c18o_prominence_factor=3.0,
                       co_verification_height_factor=3.0,
                       min_distance_vel=5.0, min_width_vel=2.0):
    """
    C18Oのピークを検出し、12COと13COの信号で検証します。
    スキャニングエフェクトなどの偽のピークを除外します。

    Args:
        c18o_spec, co12_spec, co13_spec (np.array): 各分子輝線の1Dスペクトル。
        vaxis (np.array): 速度軸。
        dv (float): 速度分解能。
        (その他の引数はピーク検出と検証のパラメータ)

    Returns:
        tuple: 検証されたピーク情報 (v_peak, t_peak, fwhm_vel, peak_idx) または (None, None, None, None)
    """
    # 1. C18Oスペクトルからピーク候補を検出
    # 平滑化とノイズ推定
    smoothed_c18o = gaussian_filter1d(c18o_spec, sigma=smoothing_sigma)
    valid_c18o = smoothed_c18o[~np.isnan(smoothed_c18o)]
    if len(valid_c18o) == 0:
        return None, None, None, None
    _, _, c18o_noise_rms = sigma_clipped_stats(valid_c18o, sigma=3.0)

    # パラメータ設定
    height_thr = c18o_height_factor * c18o_noise_rms
    prominence_thr = c18o_prominence_factor * c18o_noise_rms
    min_dist_ch = int(min_distance_vel / dv)
    min_width_ch = int(min_width_vel / dv)

    # ピーク検出と幅によるフィルタリング
    c18o_peaks, _ = find_peaks(smoothed_c18o, height=height_thr, prominence=prominence_thr, distance=min_dist_ch)

    if not len(c18o_peaks):
        return None, None, None, None
    widths, _, _, _ = peak_widths(smoothed_c18o, c18o_peaks, rel_height=0.5)
    c18o_peaks = c18o_peaks[np.where(widths >= min_width_ch)[0]]

    if not len(c18o_peaks):
        return None, None, None, None

    # 2. 検証用のCOスペクトルを準備
    smoothed_co12 = gaussian_filter1d(co12_spec, sigma=smoothing_sigma)
    _, _, co12_noise_rms = sigma_clipped_stats(smoothed_co12[~np.isnan(smoothed_co12)], sigma=3.0)
    smoothed_co13 = gaussian_filter1d(co13_spec, sigma=smoothing_sigma)
    _, _, co13_noise_rms = sigma_clipped_stats(smoothed_co13[~np.isnan(smoothed_co13)], sigma=3.0)

    # 3. C18Oの各ピーク候補を検証
    verified_peaks = []
    for peak_ch in c18o_peaks:
        # 同じ速度チャンネルで12COと13COの信号が存在するかチェック
        has_co12_signal = _check_signal_at_channel(smoothed_co12, peak_ch, co12_noise_rms, co_verification_height_factor)
        has_co13_signal = _check_signal_at_channel(smoothed_co13, peak_ch, co13_noise_rms, co_verification_height_factor)

        if has_co12_signal and has_co13_signal:
            verified_peaks.append(peak_ch)

    # 4. 検証済みピークの中から最適なものを選択
    if not len(verified_peaks):
        return None, None, None, None

    # 検証済みピークの中で、C18Oの輝度が最も高いものを採用
    main_peak_channel = verified_peaks[np.argmax(smoothed_c18o[verified_peaks])]
    v_peak = vaxis[main_peak_channel]
    t_peak = smoothed_c18o[main_peak_channel]

    # FWHMを計算
    results_fwhm = peak_widths(smoothed_c18o, [main_peak_channel], rel_height=0.5)
    fwhm_chan = results_fwhm[0][0]
    fwhm_vel = fwhm_chan * dv

    return v_peak, t_peak, fwhm_vel, main_peak_channel



def _detect_and_characterize_peak(spectrum_data, vaxis, dv,
                                  smoothing_sigma=1.0,
                                  height_factor=5.0,
                                  prominence_factor=3.0,
                                  min_distance_vel=5.0,
                                  min_width_vel=2.0):
    """
    単一のスペクトルから最も顕著なピークを検出し、その特性を返します。
    主に13COへのフォールバック時に使用します。

    Args:
        spectrum_data (np.array): 1Dのスペクトル強度配列。
        vaxis (np.array): 速度軸の配列。
        dv (float): 速度チャンネルの幅 (km/s)。
        (その他の引数はピーク検出のパラメータ)

    Returns:
        tuple: (v_peak, t_peak, fwhm_vel, peak_idx) または (None, None, None, None)
    """
    # データを平滑化
    smoothed_data = gaussian_filter1d(spectrum_data, sigma=smoothing_sigma)

    valid_data = smoothed_data[~np.isnan(smoothed_data)]
    if len(valid_data) == 0:
        return None, None, None, None

    # ノイズRMSをロバストに推定
    _, _, noise_rms = sigma_clipped_stats(valid_data, sigma=3.0)

    # ピーク検出パラメータを設定
    height_threshold = height_factor * noise_rms
    prominence_threshold = prominence_factor * noise_rms
    min_distance_channels = int(min_distance_vel / dv)
    min_width_channels = int(min_width_vel / dv)

    # ピーク検出を実行
    peaks, _ = find_peaks(
        smoothed_data,
        height=height_threshold,
        prominence=prominence_threshold,
        distance=min_distance_channels
    )

    if not len(peaks):
        return None, None, None, None

    # ピーク幅でフィルタリング
    widths, _, _, _ = peak_widths(smoothed_data, peaks, rel_height=0.5)
    valid_peaks_indices = np.where(widths >= min_width_channels)[0]

    if not len(valid_peaks_indices):
        return None, None, None, None

    filtered_peaks = peaks[valid_peaks_indices]

    # 最も強度の高いピークを選択
    main_peak_idx_in_filtered = np.argmax(smoothed_data[filtered_peaks])
    main_peak_channel = filtered_peaks[main_peak_idx_in_filtered]

    v_peak = vaxis[main_peak_channel]
    t_peak = smoothed_data[main_peak_channel]

    # FWHMを計算
    results_fwhm = peak_widths(smoothed_data, [main_peak_channel], rel_height=0.5)
    fwhm_chan = results_fwhm[0][0]
    fwhm_vel = fwhm_chan * dv

    return v_peak, t_peak, fwhm_vel, main_peak_channel



def find_velocity_from_catalog(ra_center, dec_center, bub_velocity_table, 
                              search_radius=0.1):
    """
    バブル速度カタログから該当する速度情報を検索します。
    
    Args:
        ra_center (float): バブルの中心RA（銀経）
        dec_center (float): バブルの中心Dec（銀緯）
        bub_velocity_table (pd.DataFrame): バブル速度カタログ
        search_radius (float): 検索半径（度）
        
    Returns:
        tuple: (v_peak, fwhm_vel, catalog_info) または (None, None, None)
    """
    # 座標の距離計算（簡易版）
    distances = np.sqrt((bub_velocity_table['GLON'] - ra_center)**2 + 
                       (bub_velocity_table['GLAT'] - dec_center)**2)
    
    # 閾値内のエントリを検索
    matches = bub_velocity_table[distances < search_radius]
    
    if len(matches) > 0:
        # 最も近いエントリを選択
        closest_idx = distances[matches.index].idxmin()
        closest_entry = bub_velocity_table.loc[closest_idx]
        
        # VHIIが有効な値かチェック
        if not np.isnan(closest_entry['VHII']):
            v_peak = closest_entry['VHII']
            # FWHMの情報がカタログにない場合は、デフォルト値を使用
            # D0列がある場合はそれを使用、なければデフォルト値
            if 'D0' in closest_entry and not np.isnan(closest_entry['D0']):
                # D0を速度幅の推定値として使用（適切な変換係数を適用）
                fwhm_vel = 5.0  # または適切な計算式
            else:
                fwhm_vel = 5.0  # デフォルト値（km/s）
            
            catalog_info = {
                'MWP': closest_entry['MWP'],
                'GLON': closest_entry['GLON'],
                'GLAT': closest_entry['GLAT'],
                'distance': distances[closest_idx],
                'catalog_index': closest_idx,  # カタログの行番号
                'VHII': v_peak
            }
            
            return v_peak, fwhm_vel, catalog_info
    
    return None, None, None


def _check_signal_at_channel(spectrum_data, channel, noise_rms, height_factor=3.0):
    """
    与えられたチャンネルに有意な信号が存在するかどうかを確認します。

    Args:
        spectrum_data (np.array): スペクトル強度データ。
        channel (int): 確認するチャンネルのインデックス。
        noise_rms (float): スペクトルのノイズRMS。
        height_factor (float): ノイズRMSに対する信号強度の閾値（係数）。

    Returns:
        bool: 有意な信号が存在すればTrue。
    """
    if channel < 0 or channel >= len(spectrum_data):
        return False
    # 指定されたチャンネルの強度がノイズのN倍より大きいか確認
    return spectrum_data[channel] > height_factor * noise_rms



def gaussian_filter(data, mode="valid"):#三次元データでも二次元データでも一層ずつガウシアンフィルター
    #ガウシアンフィルターの定義
    gaussian_num = [1, 4, 6, 4, 1]
    gaussian_filter = np.outer(gaussian_num, gaussian_num)
    gaussian_filter2 = gaussian_filter/np.sum(gaussian_filter)
    
    if len(data.shape) == 3:
        gau_map_list = []
        for i in range(len(data)):
            gau = fftconvolve(data[i], gaussian_filter2, mode=mode)
            gau_map_list.append(gau)
        gau_map = np.stack(gau_map_list, axis=0)

        return gau_map
    
    elif len(data.shape) == 2:
        gau_map = fftconvolve(data, gaussian_filter2, mode=mode)

        return gau_map
    
    else:
        print("shape of data must be 2 or 3")