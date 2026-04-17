"""Verifies the Thresholder raises CarmenAnalysisError on unknown method."""

from __future__ import annotations

import pandas as pd
import pytest

from carmen_analysis.errors import CarmenAnalysisError
from carmen_analysis.qc.threshold import Thresholder


def test_unknown_method_raises():
    t = Thresholder()
    df = pd.DataFrame({"assay1": [0.1, 0.2]}, index=["NTC_1", "sample_1"])
    with pytest.raises(CarmenAnalysisError):
        t.raw_thresholder(["assay1"], pd.DataFrame(), df, "bogus_method")
