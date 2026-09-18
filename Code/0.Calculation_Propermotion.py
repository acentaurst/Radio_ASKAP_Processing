"""Propagate catalogue sky coordinates to the epoch of each ASKAP observation."""

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning


def main() -> None:
    # 测试目标: Proxima Cen
    print("=== 开始计算 Proxima Cen 的坐标 ===")

    initial_ra = 217.3934657  # 度
    initial_dec = -62.6761821  # 度
    pm_ra = -3781.31  # mas/yr
    pm_dec = 769.766  # mas/yr
    epoch_init = 'J2015.5'  # 初始历元

    # 目标 MJD (例如: 60421.0 对应 2024-04-21)
    target_mjd_value = 60100.54414

    # 解析自行并传播坐标；本脚本只执行一次该计算，因此直接内联。
    try:
        pmra = float(pm_ra)
        pmdec = float(pm_dec)
        if not (float('-inf') < pmra < float('inf') and float('-inf') < pmdec < float('inf')):
            raise ValueError
    except (TypeError, ValueError):
        print('⚠️ [NO PROPER MOTION] coordinate calculation: missing proper-motion information; '
              'using pmra=0.000, pmdec=0.000 mas/yr. Epoch propagation continues without a reliable PM correction.')
        pmra, pmdec = 0.0, 0.0

    obstime_initial = Time(epoch_init)
    obstime_target = Time(target_mjd_value, format='mjd')
    initial_coord = SkyCoord(
        ra=initial_ra * u.deg,
        dec=initial_dec * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        frame='icrs',
        obstime=obstime_initial,
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        result_coord = initial_coord.apply_space_motion(new_obstime=obstime_target)

    print(f"--- 目标 MJD: {target_mjd_value} ---")
    print(f"初始坐标 (J2000): RA={initial_ra:.6f} deg, Dec={initial_dec:.6f} deg")
    print(f"计算后新坐标    : RA={result_coord.ra.deg:.6f} deg, Dec={result_coord.dec.deg:.6f} deg")

    # 如果你需要将其输出为特定的时分秒格式：
    print("\n格式化输出 (时分秒/度分秒):")
    print(result_coord.to_string('hmsdms'))


if __name__ == "__main__":
    main()
