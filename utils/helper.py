# helper functions for generic tasks
from utils.imu_data import IMUData
import numpy as np
import os
from skyfield.api import load, wgs84
from datetime import datetime, timedelta
import pytz
from math import inf
from astropy import coordinates as coord
from astropy.time import Time
import astropy.units as u

def get_data_configs(data_folder, file_idx):
    start_time_list = ['1710285434.4520473',
            '1710286544.4498281',
            '1710290158.7682009',
            '1710291262.7842135',
            '1710296420.6706645',
            '1710299824.358615',
            '1710302341.7380192',
            '1710308243.820163',
            '1709697074.3228533',
            '1709698773.3134246',
            '1708471197.0514207',
            ]
    imu_path_list = ['imus/imu_utctime_2024_03_12_23_35_25.txt',
                'imus/imu_utctime_2024_03_12_23_35_25.txt',
                'imus/imu_utctime_2024_03_13_00_35_37.txt',
                'imus/imu_utctime_2024_03_13_00_54_03.txt',
                'imus/imu_utctime_2024_03_13_02_20_00.txt',
                'imus/imu_utctime_2024_03_13_03_16_43.txt',
                'imus/imu_utctime_2024_03_13_03_58_41.txt',
                'imus/imu_utctime_2024_03_13_05_37_03.txt',
                'imus/imu_utctime_2024_03_06_03_50_54.txt',
                'imus/imu_utctime_2024_03_06_04_19_12.txt',
                'imus/imu_utctime_2024_02_20_23_19_38.txt',
                ]
    date_list = ["12Mar2024_23_17",
                "12Mar2024_23_35",
                "13Mar2024_00_35",
                "13Mar2024_00_54",
                "13Mar2024_02_20",
                "13Mar2024_03_17",
                "13Mar2024_03_59",
                "13Mar2024_05_37",
                "6Mar2024_03_51",
                "6Mar2024_04_19",
                "20Feb2024_23_19",
                ]
    imu_path = imu_path_list[file_idx]
    date = date_list[file_idx]
    imu = IMUData(os.path.join(data_folder, imu_path))
    imu_data = imu.read_imu_data()
    lat, lon, alt = imu.get_latlonalt_()
    lat_gt, lon_gt, alt_gt = np.mean(lat), np.mean(lon), np.mean(alt)
    orient = imu.get_attitude()
    yaw = np.mean(orient[2][-3000:-1])
    pitch = 0
    roll = 0
    start_time = float(start_time_list[file_idx])
    if file_idx != 8 and file_idx != 9:
        yaw = 0
    return start_time, date, lat_gt, lon_gt, alt_gt, yaw, pitch, roll

def aoa_sim_1_sat_loc(date, path, sat_id, t_start, t_stop, t_step, gs_lat=+40.07216262817384, gs_lon=-88.20958051030674, gs_elevation=182.69):
    satellites = load.tle_file(path) # either use path to tle or directly fetch latest tle from the url
    gs = wgs84.latlon(gs_lat, gs_lon, elevation_m=gs_elevation)

    # convert from timestamp to datetime
    d_1 = datetime.utcfromtimestamp(t_start)
    t_start = datetime(d_1.year, d_1.month, d_1.day, d_1.hour, d_1.minute, d_1.second, d_1.microsecond, tzinfo=pytz.UTC)

    d_2 = datetime.utcfromtimestamp(t_stop)
    t_stop = datetime(d_2.year, d_2.month, d_2.day, d_2.hour, d_2.minute, d_2.second, d_2.microsecond, tzinfo=pytz.UTC)

    for satellite in satellites:
        name = satellite.name
        if name == sat_id:
            _, t_list, dopp, az, el, range_, sat_loc = calc_doppler_satloc(satellite, gs, t_start, t_stop, 1/t_step)
            return t_list, az, el, range_, sat_loc, dopp

def calc_doppler_satloc(sat, gs, start_time, stop_time, fs, fc=11.325e9, min_ele=60):
    c = 299792458
    ts = load.timescale()
    t = ts.utc(start_time)
    date_time = start_time
    duration = stop_time-start_time
    n_list = np.arange(0, fs*duration.total_seconds() + 1, 1).astype(int)
    
    doppler_shift = np.empty(len(n_list))
    azimuth = np.empty(len(n_list))
    elevation = np.empty(len(n_list))
    range_ = np.empty(len(n_list))
    sat_loc = np.empty((len(n_list), 3))

    sat_ele_max = -inf
    for n in n_list:
        dist = (sat-gs).at(t)
        sat_loc[n] = sat.at(t).position.m
        sat_loc[n] = eci_to_ecef_pos(sat_loc[n], date_time, 'm')

        t = ts.utc(t.utc_datetime()+timedelta(milliseconds=1000/fs))
        date_time = date_time+timedelta(milliseconds=1000/fs)
        
        alt, az, distance = dist.altaz() 
        if alt.degrees > sat_ele_max:
            sat_ele_max = alt.degrees
        _, _, the_range, _, _, range_rate = dist.frame_latlon_and_rates(gs)
        doppler_shift[n] = -fc*range_rate.km_per_s*1000/c
        if alt.degrees < min_ele-10:
            doppler_shift[n] = np.nan
        azimuth[n] = az.degrees
        elevation[n] = alt.degrees
        range_[n] = the_range.m
        
    t_list = n_list/fs
    is_visible = sat_ele_max > min_ele

    return is_visible, t_list, doppler_shift, azimuth, elevation, range_, sat_loc

def range_estimation(start_time_unix, l_bound, u_bound, step, el, az, t, lat, lon, alt):
    """
    :param start_time_unix: start time in unix timestamp
    :param l_bound: range grid search lower bound (in meters)
    :param u_bound: range grid search upper bound (in meters)
    :param step: range grid search step size (in meters)
    :param el: elevation angle in degrees [nx1]
    :param az: azimuth angle in degrees [nx1]
    :param t: time array corresponding to the angles (in seconds) [nx1]
    :param lat: latitude of the ground station in degrees
    :param lon: longitude of the ground station in degrees
    :param alt: altitude of the ground station in meters

    :return: h_best: estimated range (in meters)
    """
    GM = 3.986004418e14
    start_time = Time(start_time_unix, format='unix')
    gs_ecef = lla_to_ecef(lat, lon, alt)
    dt = t[1] - t[0]

    # --- Coordinate Vectors (ENU) ---
    # Converting degrees to radians for trig functions
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    
    # e: East [-sin(lon), cos(lon), 0]
    e_vec = np.array([-np.sin(lon_r), np.cos(lon_r), 0])
    
    # n: North [-cos(lon)sin(lat), -sin(lon)sin(lat), cos(lat)]
    n_vec = np.array([
        -np.cos(lon_r) * np.sin(lat_r),
        -np.sin(lon_r) * np.sin(lat_r),
        np.cos(lat_r)
    ])
    
    # u: Up [cos(lon)cos(lat), sin(lon)cos(lat), sin(lat)]
    u_vec = np.array([
        np.cos(lon_r) * np.cos(lat_r),
        np.sin(lon_r) * np.cos(lat_r),
        np.sin(lat_r)
    ])
    
    # --- Prepare Data for Optimization Loop ---
    # Subsample indices
    indices = np.arange(0, len(t), 1)
    
    # Pre-calculate Az/El trig terms for the subsampled indices
    el_sub = el[indices]
    az_sub = az[indices]
    el_r_sub = np.deg2rad(el_sub)
    az_r_sub = np.deg2rad(az_sub)
    
    # Pre-calculate Time Array for Astropy
    # StartTime + (index * dt)
    # We construct a Time object containing all timestamps at once
    time_offsets = indices * dt
    # Create array of Time objects (or a single Time object with vector values)
    # Astropy handles Time + TimeDelta array efficiently
    t_vector = start_time + (time_offsets * u.s)

    # --- Optimization Loop ---
    h_try_list = np.arange(l_bound, u_bound + step, step)
    err_list = np.zeros(len(h_try_list))
    v_list = np.zeros(len(h_try_list))
    
    for jj, h_try in enumerate(h_try_list):
        
        # 1. Calculate Line of Sight (LOS) vectors for all subsampled points
        # Formula: cos(el)cos(az)n + cos(el)sin(az)e + sin(el)u
        # Shape will be (3, N_samples)
        
        term_n = (np.cos(el_r_sub) * np.cos(az_r_sub)) * n_vec[:, np.newaxis]
        term_e = (np.cos(el_r_sub) * np.sin(az_r_sub)) * e_vec[:, np.newaxis]
        term_u = np.sin(el_r_sub) * u_vec[:, np.newaxis]
        
        los = term_n + term_e + term_u
        
        # 2. Calculate Predicted ECEF
        R_earth = 6371e3  # Earth's radius in meters
        R_sat = R_earth + h_try
        ranges = -R_earth * np.sin(el_r_sub) + np.sqrt(R_sat**2 - (R_earth**2) * (np.cos(el_r_sub)**2))
        ecef_pre = gs_ecef[:, np.newaxis] + (los * ranges)
        
        # 3. Convert to ECI (Vectorized)
        # Passing (3, N) coordinates and N times
        eci_pre = ecef_to_eci_pos(ecef_pre, t_vector, units='m')
        
        # 4. Orbit Physics Checks
        # Vector norms (columns)
        orbit_radii = np.linalg.norm(eci_pre, axis=0)
        orbit_height_mean = np.mean(orbit_radii)
        
        # Velocity (Finite difference)
        # diff along axis 1 (time)
        v_diff = np.diff(eci_pre, axis=1)
        v_mean_vec = np.mean(v_diff, axis=1) / dt
        v_norm = np.linalg.norm(v_mean_vec)
        
        # 5. Error calculation: |v^2 * r - GM|
        err = abs((v_norm**2) * orbit_height_mean - GM)
        err_list[jj] = err

        v_ecef = np.diff(ecef_pre, axis=1) / dt
        v_list[jj] = np.linalg.norm(np.mean(v_ecef, axis=1))

    min_idx = np.argmin(err_list)
    h_best = h_try_list[min_idx]
    v_est = v_list[min_idx]
    return h_best, v_est

def eci_to_ecef_pos(eci_pos, epoch, units):
    # if isinstance(epoch, datetime):
    #     epoch = Time(datetime)
    if units == 'm':
        us = u.m
    else:
        us = u.km
    eci_coord = coord.CartesianRepresentation(x=eci_pos[0]*us, y=eci_pos[1]*us, z=eci_pos[2]*us)
    gcrs = coord.GCRS(eci_coord, obstime=epoch)
    itrf = gcrs.transform_to(coord.ITRS(obstime=epoch))
    ecef_pos = np.array(itrf.cartesian.xyz.value)
    return ecef_pos

def ecef_to_eci_pos(ecef_pos, epoch, units):
    """
    Converts ECEF (ITRS) coordinates to ECI (GCRS).
    
    Parameters:
    ecef_pos : array-like (x, y, z)
    epoch    : astropy.time.Time object
    units    : str ('m' or 'km')
    """
    
    # Set units based on input string
    if units == 'm':
        us = u.m
    else:
        us = u.km

    # 1. Create Cartesian Representation of the input ECEF position
    ecef_cart = coord.CartesianRepresentation(
        x=ecef_pos[0] * us, 
        y=ecef_pos[1] * us, 
        z=ecef_pos[2] * us
    )

    # 2. Create the ITRS (Earth-Fixed) object
    # Note: obstime is crucial here to define the Earth's orientation
    itrs_frame = coord.ITRS(ecef_cart, obstime=epoch)

    # 3. Transform to GCRS (Earth-Centered Inertial)
    gcrs_frame = itrs_frame.transform_to(coord.GCRS(obstime=epoch))

    # 4. Extract coordinates as a numpy array
    eci_pos = np.array(gcrs_frame.cartesian.xyz.value)
    
    return eci_pos

def lla_to_ecef(lat, lon, alt):
    """
    lat, lon in degrees. alt in meters.
    Uses the specific constants provided in the prompt.
    """
    f = 0.003352810664747
    R = 6378137.0
    
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    h = alt

    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    
    e2 = f * (2 - f)
    N = R / np.sqrt(1 - e2 * (sinphi**2))
    
    rho = (N + h) * cosphi
    z = (N * (1 - e2) + h) * sinphi
    
    x = rho * np.cos(lam)
    y = rho * np.sin(lam)
    
    # Return as 3x1 vector
    return np.array([x, y, z])