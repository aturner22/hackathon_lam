# Mapping from variable names to indices for weather model variables

var_idx = {
    "pres_0g": 0,   # Atmospheric pressure at ground level (Pa)
    "pres_0s": 1,   # Atmospheric pressure at sea level (Pa)
    "nlwrs": 2,     # Net longwave radiation flux at ground level (W/m²)
    "nswrs": 3,     # Net shortwave radiation flux at ground level (W/m²)
    "rh_2": 4,      # Relative humidity at 2 m above ground (in [0,1])
    "rh_s": 5,      # Relative humidity at lowest MEPS level (in [0,1])
    "t_2": 6,       # Instantaneous temperature at 2 m above ground (K)
    "t_s": 7,       # Instantaneous temperature at lowest MEPS level (K)
    "t_500": 8,     # Instantaneous temperature at 500 hPa pressure (K)
    "t_850": 9,     # Instantaneous temperature at 850 hPa pressure (K)
    "u_65": 10,     # u-component (east-west) of wind at 65 m above ground (m/s)
    "u_s": 11,      # u-component of wind at lowest MEPS level (m/s)
    "u_850": 12,    # u-component of wind at 850 hPa pressure (m/s)
    "v_65": 13,     # v-component (north-south) of wind at 65 m above ground (m/s)
    "v_s": 14,      # v-component of wind at lowest MEPS level (m/s)
    "v_850": 15,    # v-component of wind at 850 hPa pressure (m/s)
    "wint": 16,     # Integrated column water vapor over the full grid cell (kg/m²)
    "z_500": 17,    # Instantaneous geopotential at 500 hPa pressure (m²/s²)
    "z_1000": 18    # Instantaneous geopotential at 1000 hPa pressure (m²/s²)
}
