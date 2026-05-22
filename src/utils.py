""" Shared utility functions and constants """

TAG_INFO = {
    "FT-201": {"label": "Biogas inlet flow", "unit": "Nm3/h"},
    "FT-801": {"label": "Biomethane outlet flow", "unit": "Nm3/h"},
    "FT-901": {"label": "Exhaust gas flow", "unit": "Nm3/h"},
    "Mode": {"label": "Run mode", "unit": "integer"},
    "PT-901": {"label": "Gas line inlet vacuum pressure", "unit": "bar"},
    "PT-902": {"label": "Gas line outlet vacuum pressure", "unit": "bar"},
    "PT-903": {"label": "Vacuum pressure in oil separator", "unit": "bar"},
    "TT-901": {"label": "Oil separator gas temperature", "unit": "°C"},
    "TT-902": {"label": "Discharge 1 temperature", "unit": "°C"},
    "TT-903": {"label": "Discharge 2 temperature", "unit": "°C"},
    "TT-904": {"label": "Oil temperature", "unit": "°C"},
    "VSD-901_CORRENT": {"label": "Vacuum pump current", "unit": "A"},
    "VSD-901_POWER": {"label": "Vacuum pump power", "unit": "W"},
    "VSD-901_RPM": {"label": "Vacuum pump rotation", "unit": "RPM"},
    "VSD-901_SPEED": {"label": "Vacuum pump speed", "unit": "%"},
    "LS-901": {"label": "Oil level sensor", "unit": "true/false"},
}

# Warning / trip thresholds
THRESHOLDS = {
    "TT-901": {"warning": 135, "trip": 140},
    "TT-902": {"warning": 105, "trip": 110},
    "TT-903": {"warning": 105, "trip": 110},
    "TT-904": {"warning":  90, "trip": 110},
    "PT-903": {"warning":  0.4, "trip":  0.5},
}