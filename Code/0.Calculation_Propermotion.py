import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord


def calculate_new_epoch_coordinates(ra, dec, pm_ra_cosdec, pm_dec, initial_epoch, target_mjd, distance=None,
                                    radial_velocity=None):
    """
    计算考虑自行后的新历元坐标。

    参数:
    ra (float): 初始赤经，单位：度 (deg)
    dec (float): 初始赤纬，单位：度 (deg)
    pm_ra_cosdec (float): 赤经方向自行 (已包含 cos(dec) 因子)，单位：毫角秒/年 (mas/yr)
    pm_dec (float): 赤纬方向自行，单位：毫角秒/年 (mas/yr)
    initial_epoch (str): 初始历元，例如 'J2000.0'
    target_mjd (float): 目标时间的简化儒略日 (Modified Julian Date, MJD)
    distance (float, 可选): 目标距离，单位：秒差距 (pc)。用于更精确的3D运动计算。
    radial_velocity (float, 可选): 视向速度，单位：千米/秒 (km/s)。用于更精确的3D运动计算。

    返回:
    SkyCoord: 计算自行后在目标 MJD 历元下的坐标对象
    """

    # 1. 解析初始历元和目标 MJD
    obstime_initial = Time(initial_epoch)
    obstime_target = Time(target_mjd, format='mjd')

    # 2. 构建 kwargs 字典以处理可选参数 (距离和视向速度)
    coord_kwargs = {
        'ra': ra * u.deg,
        'dec': dec * u.deg,
        'pm_ra_cosdec': pm_ra_cosdec * u.mas / u.yr,
        'pm_dec': pm_dec * u.mas / u.yr,
        'frame': 'icrs',
        'obstime': obstime_initial
    }

    if distance is not None:
        coord_kwargs['distance'] = distance * u.pc
    if radial_velocity is not None:
        coord_kwargs['radial_velocity'] = radial_velocity * (u.km / u.s)

    # 3. 初始化 SkyCoord 对象
    initial_coord = SkyCoord(**coord_kwargs)

    # 4. 应用空间运动计算新历元坐标
    # 注意: 如果没有提供 distance 和 radial_velocity，astropy 将发出警告并假设它们为0进行近似计算
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # 忽略因缺少距离/视向速度而产生的近似计算警告
        new_coord = initial_coord.apply_space_motion(new_obstime=obstime_target)

    return new_coord

if __name__ == "__main__":
    # 测试目标: Proxima Cen
    print("=== 开始计算 Proxima Cen 的坐标 ===")

    initial_ra = 217.3934657  # 度
    initial_dec = -62.6761821  # 度
    pm_ra = -3781.31  # mas/yr
    pm_dec = 769.766  # mas/yr
    epoch_init = 'J2015.5'  # 初始历元

    # 目标 MJD (例如: 60421.0 对应 2024-04-21)
    target_mjd_value = 60100.54414

    # 调用函数 (仅提供基础自行参数)
    result_coord = calculate_new_epoch_coordinates(
        ra=initial_ra,
        dec=initial_dec,
        pm_ra_cosdec=pm_ra,
        pm_dec=pm_dec,
        initial_epoch=epoch_init,
        target_mjd=target_mjd_value
    )

    print(f"--- 目标 MJD: {target_mjd_value} ---")
    print(f"初始坐标 (J2000): RA={initial_ra:.6f} deg, Dec={initial_dec:.6f} deg")
    print(f"计算后新坐标    : RA={result_coord.ra.deg:.6f} deg, Dec={result_coord.dec.deg:.6f} deg")

    # 如果你需要将其输出为特定的时分秒格式：
    print("\n格式化输出 (时分秒/度分秒):")
    print(result_coord.to_string('hmsdms'))