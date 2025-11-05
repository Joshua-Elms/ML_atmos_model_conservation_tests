SUPPORTED_MODELS = {
    "SFNO",
    "Pangu6",
    "Pangu6x",
    "Pangu24",
    "GraphCastOperational",
    "FuXi",
    "FuXiShort",
    "FuXiMedium",
    "FuXiLong",
    "FCN3",
    "FCN",
}

STANDARD_13_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

SL = 0 # single-level, often called "surface"
PL = 1 # pressure levels
IN = 2 # invariant variable, e.g. land-sea mask

SL_VARIABLES = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "msl", "tcwv", "tp06"]
PL_VARIABLES = [f"{var}{level}" for var in ["u", "v", "t", "q", "r", "w", "z"] for level in STANDARD_13_LEVELS]
IN_VARIABLES = ["z", "lsm"]

MASTER_VARIABLES_NAMES =  SL_VARIABLES + PL_VARIABLES + IN_VARIABLES
MASTER_VARIABLES_TYPES = [SL for _ in SL_VARIABLES] + [PL for _ in PL_VARIABLES] + [IN for _ in IN_VARIABLES]

MODEL_VARIABLES = dict(
    FCN3 = dict(
        names = [
            "u10m", "v10m", "u100m", "v100m", "t2m",
            "msl", "tcwv", "u50", "u100", "u150", "u200",
            "u250", "u300", "u400", "u500", "u600", "u700",
            "u850", "u925", "u1000", "v50", "v100", "v150",
            "v200", "v250", "v300", "v400", "v500", "v600",
            "v700", "v850", "v925", "v1000", "z50", "z100",
            "z150", "z200", "z250", "z300", "z400", "z500",
            "z600", "z700", "z850", "z925", "z1000", "t50",
            "t100", "t150", "t200", "t250", "t300", "t400",
            "t500", "t600", "t700", "t850", "t925", "t1000",
            "q50", "q100", "q150", "q200", "q250", "q300",
            "q400", "q500", "q600", "q700", "q850", "q925",
            "q1000",
        ],
        types = [SL for _ in range(7)] + [PL for _ in range(65)]
    ),
    FCN = dict(
        names = [
            "u10m", "v10m", "t2m", "sp", "msl", "t850",
            "u1000", "v1000", "z1000", "u850", "v850",
            "z850", "u500", "v500", "z500", "t500",
            "z50", "r500", "r850", "tcwv", "u100m",
            "v100m", "u250", "v250", "z250", "t250",
        ],
        types = [SL for _ in range(5)] + [PL for _ in range(14)] + [SL for _ in range(3)] + [PL for _ in range(4)]
    ),
    FuXi = dict(
        names = [
            "z50","z100","z150","z200","z250","z300",
            "z400","z500","z600","z700","z850","z925",
            "z1000","t50","t100","t150","t200","t250",
            "t300","t400","t500","t600","t700","t850",
            "t925","t1000","u50","u100","u150","u200",
            "u250","u300","u400","u500","u600","u700",
            "u850","u925","u1000","v50","v100","v150",
            "v200","v250","v300","v400","v500","v600",
            "v700","v850","v925","v1000","r50","r100",
            "r150","r200","r250","r300","r400","r500",
            "r600","r700","r850","r925","r1000","t2m",
            "u10m","v10m","msl","tp06",
        ],  
        types = [PL for _ in range(65)] + [SL for _ in range(5)]
        ),
    SFNO = dict(
        names = [
            "u10m", "v10m", "u100m", "v100m", "t2m", "sp",
            "msl", "tcwv", "u50", "u100", "u150", "u200",
            "u250", "u300", "u400", "u500", "u600", "u700",
            "u850", "u925", "u1000", "v50", "v100", "v150",
            "v200", "v250", "v300", "v400", "v500", "v600",
            "v700", "v850", "v925", "v1000", "z50", "z100",
            "z150", "z200", "z250", "z300", "z400", "z500",
            "z600", "z700", "z850", "z925", "z1000", "t50",
            "t100", "t150", "t200", "t250", "t300", "t400",
            "t500", "t600", "t700", "t850", "t925", "t1000",
            "q50", "q100", "q150", "q200", "q250", "q300",
            "q400", "q500", "q600", "q700", "q850", "q925",
            "q1000",
        ],
        types = [SL for _ in range(8)] + [PL for _ in range(65)]
    ),
    Pangu6 = dict(
        names = [
            "z1000", "z925", "z850", "z700", "z600", "z500",
            "z400", "z300", "z250", "z200", "z150", "z100", 
            "z50", "q1000", "q925", "q850", "q700", "q600", 
            "q500", "q400", "q300", "q250", "q200", "q150", 
            "q100", "q50", "t1000", "t925", "t850", "t700", 
            "t600", "t500", "t400", "t300", "t250", "t200", 
            "t150", "t100", "t50", "u1000", "u925", "u850", 
            "u700", "u600", "u500", "u400", "u300", "u250", 
            "u200", "u150", "u100", "u50", "v1000", "v925", 
            "v850", "v700", "v600", "v500", "v400", "v300", 
            "v250", "v200", "v150", "v100", "v50", "msl", 
            "u10m", "v10m", "t2m",
        ],
        types = [PL for _ in range(65)] + [SL for _ in range(4)]
    ),
    GraphCastOperational = dict(
        names = [ 
            "t2m", "msl", "u10m", "v10m", "tp06", "t50", 
            "t100", "t150", "t200", "t250", "t300", "t400", 
            "t500", "t600", "t700", "t850", "t925", "t1000", 
            "z50", "z100", "z150", "z200", "z250", "z300", 
            "z400", "z500", "z600", "z700", "z850", "z925", 
            "z1000", "u50", "u100", "u150", "u200", "u250", 
            "u300", "u400", "u500", "u600", "u700", "u850", 
            "u925", "u1000", "v50", "v100", "v150", "v200", 
            "v250", "v300", "v400", "v500", "v600", "v700", 
            "v850", "v925", "v1000", "w50", "w100", "w150", 
            "w200", "w250", "w300", "w400", "w500", "w600", 
            "w700", "w850", "w925", "w1000", "q50", "q100", 
            "q150", "q200", "q250", "q300", "q400", "q500", 
            "q600", "q700", "q850", "q925", "q1000", "z", "lsm",
        ],
        types = [SL for _ in range(5)] + [PL for _ in range(78)] + [IN, IN]
    ),
)

# these are just alternative versions of the same models
MODEL_VARIABLES["FuXiShort"] = MODEL_VARIABLES["FuXi"]
MODEL_VARIABLES["FuXiMedium"] = MODEL_VARIABLES["FuXi"]
MODEL_VARIABLES["FuXiLong"] = MODEL_VARIABLES["FuXi"]
MODEL_VARIABLES["Pangu6x"] = MODEL_VARIABLES["Pangu6"]
MODEL_VARIABLES["Pangu24"] = MODEL_VARIABLES["Pangu6"]

# these should be found in model input_coords here: /N/slate/jmelms/projects/earth2studio-cu126/earth2studio/models/px/sfno.py
MODEL_LATITUDE_ORDERING = dict(
    SFNO = "descending",                # should be descending
    FCN3 = "descending",                # should be descending
    FCN = "descending",                 # should be descending
    Pangu6 = "descending",              # should be descending
    Pangu6x = "descending",             # should be descending
    Pangu24 = "descending",             # should be descending
    FuXi = "descending",                # should be descending
    FuXiShort = "descending",           # should be descending
    FuXiMedium = "descending",          # should be descending
    FuXiLong = "descending",            # should be descending
    GraphCastOperational = "ascending", # should be ascending
)

CDS_TO_E2S = {
        "geopotential": "z",
        "relative_humidity": "r",
        "specific_humidity": "q",
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "vertical_velocity": "w",
        "10m_u_component_of_wind": "u10m",
        "10m_v_component_of_wind": "v10m",
        "2m_temperature": "t2m",
        "mean_sea_level_pressure": "msl",
        "surface_pressure": "sp",
        "100m_u_component_of_wind": "u100m",
        "100m_v_component_of_wind": "v100m",
        "total_column_water_vapour": "tcwv",
        "total_precipitation_06": "tp06",
        "land_sea_mask": "lsm",
        "total_precipitation": "tp",
    }

E2S_TO_CDS = {v: k for k, v in CDS_TO_E2S.items()}

MODEL_HISTORY_STEPS = {
    "SFNO": 1,
    "Pangu6": 1,
    "Pangu6x": 1,
    "Pangu24": 1,
    "GraphCastOperational": 2,
    "FuXi": 2,
    "FuXiShort": 2,
    "FuXiMedium": 2,
    "FuXiLong": 2,
    "FCN3": 1,
    "FCN": 1,
}

MODEL_TIME_STEP_HOURS = {
    "SFNO": 6,
    "Pangu6": 6,
    "Pangu6x": 6,
    "Pangu24": 24,
    "GraphCastOperational": 6,
    "FuXi": 6,
    "FuXiShort": 6,
    "FuXiMedium": 6,
    "FuXiLong": 6,
    "FCN3": 6,
    "FCN": 6,
    }

if __name__=="__main__":
    # following is just a useful check to make sure lists are of same length
    for m, v in MODEL_VARIABLES.items():
        print(f"{m} has {len(v['names'])=} and {len(v['types'])=}")