import pandas as pd

from scripts.calibrate_battledim import scada_flow_to_lps, wntr_flow_to_lps


def test_battledim_and_wntr_flow_units_are_converted_to_lps():
    index = pd.Index(["sample"])
    assert scada_flow_to_lps(pd.DataFrame({"meter": [36.0]}, index=index)).loc["sample", "meter"] == 10.0
    assert wntr_flow_to_lps(pd.DataFrame({"link": [0.01]}, index=index)).loc["sample", "link"] == 10.0
