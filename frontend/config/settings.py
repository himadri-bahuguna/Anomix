from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_TITLE = "HMC 30W DC-DC Converter"
APP_SUBTITLE = "Reliability & Predictive Monitoring"
VERSION = "MVP 0.1"

PARAMETERS = {
    "output_voltage_pos15_V": "+15V Output",
    "output_voltage_neg15_V": "-15V Output",
    "output_voltage_5V_V": "5V Output",
    "efficiency_percent": "Efficiency",
    "switching_frequency_Hz": "Switching Frequency",
    "regulation_percent": "Regulation",
    "ripple_mV": "Ripple",
    "turn_on_rise_time_us": "Turn-On Rise Time",
}
CHECKPOINTS = [0, 24, 96, 168]
