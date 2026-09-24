"""
Quality Control (QC) and Reconciliation Engine for India-WRIS / CWC River Level Data.
Tailored for Jia Bharali (N.T. Road Crossing, Sonitpur) and extensible across CWC gauge networks.

Pipeline Flow:
India-WRIS/CWC Telemetry + Manual -> QC Checks -> Reconciliation -> Observed River Level -> Features / ML -> Dashboard.

Core Principles:
1. Prefer fresh, validated Telemetry for real-time inference (low latency).
2. Fall back to verified Manual staff-gauge observations when Telemetry is offline, stale, or anomalous.
3. Reject sensor noise/spikes without suppressing genuine rapid flood rises.
4. Mark interpolated values as DERIVED — INTERPOLATED (never treat as observed ground truth).
5. If neither source is valid, return UNAVAILABLE (never zero-fill or substitute GloFAS discharge).
6. GloFAS discharge remains strictly separate as MODELLED in m³/s.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants & Provenance Taxonomy
# ---------------------------------------------------------------------------

PROV_OBS_TELEMETRY = "OBSERVED — TELEMETRY"
PROV_OBS_MANUAL = "OBSERVED — MANUAL"
PROV_DERIVED_INTERPOLATED = "DERIVED — INTERPOLATED"
PROV_DEMO_TELEMETRY = "DEMO / SIMULATED — TELEMETRY"
PROV_DEMO_MANUAL = "DEMO / SIMULATED — MANUAL"
PROV_UNAVAILABLE = "UNAVAILABLE"

CAT_OBSERVED = "OBSERVED"
CAT_DERIVED = "DERIVED"
CAT_DEMO = "DEMO / SIMULATED"
CAT_UNAVAILABLE = "UNAVAILABLE"

QC_PASSED = "PASSED_QC"
QC_OUT_OF_BOUNDS = "FAILED_QC_OUT_OF_BOUNDS"
QC_SPIKE = "FAILED_QC_SPIKE"
QC_RATE_EXCEEDED = "FAILED_QC_RATE_OF_CHANGE"
QC_FLATLINE = "FAILED_QC_STUCK_SENSOR"
QC_MISSING = "MISSING"
QC_STALE = "STALE"
FLAG_FALLBACK_MANUAL = "FALLBACK_MANUAL"
FLAG_INTERPOLATED = "INTERPOLATED"

_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Station QC Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StationQCConfig:
    """Configurable physical limits and QC thresholds for a gauge station."""
    station_code: str = settings.PRIMARY_STATION_CODE
    station_name: str = settings.PRIMARY_STATION_NAME
    river_name: str = settings.PRIMARY_RIVER_NAME
    danger_level_m: float = settings.PRIMARY_DANGER_LEVEL_M
    warning_level_m: float = settings.PRIMARY_WARNING_LEVEL_M
    hfl_m: float = settings.PRIMARY_HFL_M
    min_stage_m: float = settings.PRIMARY_MIN_STAGE_M
    max_stage_m: float = settings.PRIMARY_MAX_STAGE_M
    max_rate_of_change_m_per_hr: float = settings.PRIMARY_MAX_RATE_OF_CHANGE_M_PER_HR
    max_spike_deviation_m: float = settings.PRIMARY_MAX_SPIKE_DEVIATION_M
    max_flatline_hours: int = settings.PRIMARY_MAX_FLATLINE_HOURS
    telemetry_max_age_hours: float = getattr(settings, "WRIS_TELEMETRY_MAX_AGE_HOURS", 3.0)
    manual_max_age_hours: float = getattr(settings, "WRIS_MANUAL_MAX_AGE_HOURS", 24.0)

    @classmethod
    def for_primary(cls) -> StationQCConfig:
        """Returns StationQCConfig for Brahmaputra at Tezpur (Primary)."""
        return cls(
            station_code=settings.PRIMARY_STATION_CODE,
            station_name=settings.PRIMARY_STATION_NAME,
            river_name=settings.PRIMARY_RIVER_NAME,
            danger_level_m=settings.PRIMARY_DANGER_LEVEL_M,
            warning_level_m=settings.PRIMARY_WARNING_LEVEL_M,
            hfl_m=settings.PRIMARY_HFL_M,
            min_stage_m=settings.PRIMARY_MIN_STAGE_M,
            max_stage_m=settings.PRIMARY_MAX_STAGE_M,
            max_rate_of_change_m_per_hr=settings.PRIMARY_MAX_RATE_OF_CHANGE_M_PER_HR,
            max_spike_deviation_m=settings.PRIMARY_MAX_SPIKE_DEVIATION_M,
            max_flatline_hours=settings.PRIMARY_MAX_FLATLINE_HOURS,
        )

    @classmethod
    def for_secondary(cls) -> StationQCConfig:
        """Returns StationQCConfig for Jia Bharali at N.T. Road Crossing (Secondary)."""
        return cls(
            station_code=settings.SECONDARY_STATION_CODE,
            station_name=settings.SECONDARY_STATION_NAME,
            river_name=settings.SECONDARY_RIVER_NAME,
            danger_level_m=settings.SECONDARY_DANGER_LEVEL_M,
            warning_level_m=settings.SECONDARY_WARNING_LEVEL_M,
            hfl_m=settings.SECONDARY_HFL_M,
            min_stage_m=settings.SECONDARY_MIN_STAGE_M,
            max_stage_m=settings.SECONDARY_MAX_STAGE_M,
            max_rate_of_change_m_per_hr=settings.SECONDARY_MAX_RATE_OF_CHANGE_M_PER_HR,
            max_spike_deviation_m=settings.SECONDARY_MAX_SPIKE_DEVIATION_M,
            max_flatline_hours=settings.SECONDARY_MAX_FLATLINE_HOURS,
        )

    @classmethod
    def from_settings(cls, station: str = "primary") -> StationQCConfig:
        if station.lower() in ("secondary", "jia bharali", "jiabharali", "nt road crossing", "12359"):
            return cls.for_secondary()
        return cls.for_primary()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StationQCConfig:
        return cls(
            station_code=str(d.get("station_code", settings.PRIMARY_STATION_CODE)),
            station_name=str(d.get("station_name", settings.PRIMARY_STATION_NAME)),
            river_name=str(d.get("river", settings.PRIMARY_RIVER_NAME)),
            danger_level_m=float(d.get("danger_level_m", settings.PRIMARY_DANGER_LEVEL_M)),
            warning_level_m=float(d.get("warning_level_m", settings.PRIMARY_WARNING_LEVEL_M)),
            hfl_m=float(d.get("hfl_m", settings.PRIMARY_HFL_M)),
            min_stage_m=float(d.get("min_stage_m", settings.PRIMARY_MIN_STAGE_M)),
            max_stage_m=float(d.get("max_stage_m", settings.PRIMARY_MAX_STAGE_M)),
            max_rate_of_change_m_per_hr=float(d.get("max_rate_of_change_m_per_hr", settings.PRIMARY_MAX_RATE_OF_CHANGE_M_PER_HR)),
            max_spike_deviation_m=float(d.get("max_spike_deviation_m", settings.PRIMARY_MAX_SPIKE_DEVIATION_M)),
            max_flatline_hours=int(d.get("max_flatline_hours", settings.PRIMARY_MAX_FLATLINE_HOURS)),
        )


# ---------------------------------------------------------------------------
# River QC Checker
# ---------------------------------------------------------------------------

class RiverQCChecker:
    """Performs physical plausibility, rate of rise, spike, and flatline validation."""

    def __init__(self, config: StationQCConfig | None = None) -> None:
        self.config = config or StationQCConfig.from_settings()

    def check_range(self, level_m: float | None) -> tuple[bool, str]:
        """Validates that water level is within physical datum bounds for this station."""
        if level_m is None or pd.isna(level_m):
            return False, QC_MISSING
        if not (self.config.min_stage_m <= float(level_m) <= self.config.max_stage_m):
            return False, QC_OUT_OF_BOUNDS
        return True, QC_PASSED

    def check_rate_of_change(
        self,
        current_level_m: float | None,
        prev_level_m: float | None,
        time_diff_hours: float,
    ) -> tuple[bool, str]:
        """
        Validates rate of rise/fall against max permissible physical change.
        Accommodates genuine rapid flood rises (up to max_rate_of_change_m_per_hr).
        """
        if current_level_m is None or prev_level_m is None:
            return True, QC_PASSED
        if time_diff_hours <= 0:
            return True, QC_PASSED

        rate = abs(current_level_m - prev_level_m) / time_diff_hours
        if rate > self.config.max_rate_of_change_m_per_hr:
            return False, QC_RATE_EXCEEDED
        return True, QC_PASSED

    def check_observation(
        self,
        level_m: float | None,
        prev_level_m: float | None = None,
        time_diff_hours: float = 1.0,
        age_minutes: float | None = None,
        max_age_hours: float | None = None,
    ) -> tuple[bool, str]:
        """Validates an isolated or streaming observation."""
        ok_range, flag_range = self.check_range(level_m)
        if not ok_range:
            return False, flag_range

        if max_age_hours is not None and age_minutes is not None:
            if age_minutes > (max_age_hours * 60.0):
                return False, QC_STALE

        if prev_level_m is not None:
            ok_rate, flag_rate = self.check_rate_of_change(
                level_m, prev_level_m, time_diff_hours
            )
            if not ok_rate:
                return False, flag_rate

        return True, QC_PASSED

    def filter_series(self, series: pd.Series) -> pd.DataFrame:
        """
        Validates an entire time series.
        Distinguishes genuine rapid flood rises from single-point transient spikes:
        - Spike: jumps by > max_spike_deviation_m and immediately drops back in next reading.
        - Genuine flood rise: sustained elevation or continuing rise across consecutive readings.
        """
        if series.empty:
            return pd.DataFrame(columns=["level_m", "is_valid", "qc_flag"])

        s = series.sort_index().astype(float)
        n = len(s)
        levels = s.values
        times = s.index

        is_valid = np.ones(n, dtype=bool)
        flags = [QC_PASSED] * n

        # 1. Range test
        for i in range(n):
            val = levels[i]
            if pd.isna(val):
                is_valid[i] = False
                flags[i] = QC_MISSING
            elif not (self.config.min_stage_m <= val <= self.config.max_stage_m):
                is_valid[i] = False
                flags[i] = QC_OUT_OF_BOUNDS

        # 2. Rate-of-change and Spike test
        for i in range(1, n):
            if not is_valid[i] or not is_valid[i - 1]:
                continue
            dt_hours = (times[i] - times[i - 1]).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue

            diff_prev = levels[i] - levels[i - 1]
            rate = abs(diff_prev) / dt_hours

            # Check if isolated single-point spike (reverting in next step)
            is_spike = False
            if i + 1 < n and is_valid[i + 1]:
                diff_next = levels[i + 1] - levels[i]
                dt_next = (times[i + 1] - times[i]).total_seconds() / 3600.0
                if dt_next > 0:
                    # Opposite direction jump returning close to base
                    if (
                        abs(diff_prev) > self.config.max_spike_deviation_m
                        and abs(diff_next) > self.config.max_spike_deviation_m
                        and (diff_prev * diff_next < 0)  # opposite signs
                        and abs(levels[i + 1] - levels[i - 1]) < 0.40  # returns to pre-spike level
                    ):
                        is_spike = True

            if is_spike:
                is_valid[i] = False
                flags[i] = QC_SPIKE
            elif rate > self.config.max_rate_of_change_m_per_hr:
                # If not a confirmed sustained trend, flag rate exceeded
                is_valid[i] = False
                flags[i] = QC_RATE_EXCEEDED

        # 3. Flatline / stuck sensor test (e.g. 24h consecutive identical float value)
        # Note: only apply if stage is not at normal calm baseflow
        if n >= self.config.max_flatline_hours:
            run_len = 1
            for i in range(1, n):
                if (
                    not pd.isna(levels[i])
                    and not pd.isna(levels[i - 1])
                    and abs(levels[i] - levels[i - 1]) < 1e-4
                ):
                    run_len += 1
                    if run_len >= self.config.max_flatline_hours and levels[i] > (self.config.min_stage_m + 0.5):
                        is_valid[i] = False
                        flags[i] = QC_FLATLINE
                else:
                    run_len = 1

        result_df = pd.DataFrame(
            {
                "level_m": levels,
                "is_valid": is_valid,
                "qc_flag": flags,
            },
            index=times,
        )
        return result_df


# ---------------------------------------------------------------------------
# River Data Reconciler
# ---------------------------------------------------------------------------

class RiverDataReconciler:
    """
    Reconciles Telemetry and Manual observation streams.
    - Real-time: Telemetry preferred -> Manual fallback -> UNAVAILABLE.
    - Historical: Merges hourly grid with explicit provenance flags and DERIVED interpolation.
    """

    def __init__(self, config: StationQCConfig | None = None) -> None:
        self.config = config or StationQCConfig.from_settings()
        self.checker = RiverQCChecker(self.config)

    def reconcile_latest(
        self,
        telemetry_obs: dict[str, Any] | None,
        manual_obs: dict[str, Any] | None,
        is_mock: bool = False,
    ) -> dict[str, Any]:
        """
        Reconciles the latest single reading for real-time inference.
        Rules:
        1. Validate Telemetry: fresh, within bounds, passes QC. If OK -> use TELEMETRY.
        2. If Telemetry invalid/stale/missing -> Validate Manual. If OK -> use MANUAL fallback.
        3. If both invalid/missing -> return UNAVAILABLE with None level (never 0.0 or GloFAS).
        """
        now = datetime.datetime.now(_IST)

        # 1. Evaluate Telemetry stream
        t_available = False
        t_valid = False
        t_level = None
        t_age = None
        t_flag = QC_MISSING

        if telemetry_obs and telemetry_obs.get("observed_level_m") is not None:
            t_available = True
            t_level = float(telemetry_obs["observed_level_m"])
            t_age = telemetry_obs.get("data_age_minutes")
            t_valid, t_flag = self.checker.check_observation(
                level_m=t_level,
                prev_level_m=telemetry_obs.get("prev_level_m"),
                time_diff_hours=float(telemetry_obs.get("time_diff_hours", 1.0)),
                age_minutes=t_age,
                max_age_hours=self.config.telemetry_max_age_hours,
            )

        # 2. Evaluate Manual stream
        m_available = False
        m_valid = False
        m_level = None
        m_age = None
        m_flag = QC_MISSING

        if manual_obs and manual_obs.get("observed_level_m") is not None:
            m_available = True
            m_level = float(manual_obs["observed_level_m"])
            m_age = manual_obs.get("data_age_minutes")
            m_valid, m_flag = self.checker.check_observation(
                level_m=m_level,
                prev_level_m=manual_obs.get("prev_level_m"),
                time_diff_hours=float(manual_obs.get("time_diff_hours", 1.0)),
                age_minutes=m_age,
                max_age_hours=self.config.manual_max_age_hours,
            )

        telemetry_status = {
            "available": t_available,
            "valid": t_valid,
            "level_m": t_level,
            "data_age_minutes": t_age,
            "qc_flag": t_flag,
        }
        manual_status = {
            "available": m_available,
            "valid": m_valid,
            "level_m": m_level,
            "data_age_minutes": m_age,
            "qc_flag": m_flag,
        }

        # 3. Decision Logic
        if t_valid:
            # Primary: Telemetry
            data_cat = CAT_DEMO if is_mock else CAT_OBSERVED
            prov = PROV_DEMO_TELEMETRY if is_mock else PROV_OBS_TELEMETRY
            reconciled = {
                "observed_level_m": t_level,
                "timestamp": telemetry_obs.get("timestamp"),
                "data_category": data_cat,
                "source": "India-WRIS/CWC Telemetry",
                "provenance": prov,
                "unit": "m",
                "station_code": self.config.station_code,
                "station_name": self.config.station_name,
                "river": self.config.river_name,
                "danger_level_m": self.config.danger_level_m,
                "hfl_m": self.config.hfl_m,
                "retrieved_at": now.isoformat(),
                "data_age_minutes": t_age,
                "detail": f"Active: Validated telemetry observation ({t_flag}).",
                "reconciliation_method": "TELEMETRY_PRIMARY",
                "source_stream": "TELEMETRY",
                "is_fallback_active": False,
                "qc_flag": t_flag,
                "telemetry_status": telemetry_status,
                "manual_status": manual_status,
                "timeseries": telemetry_obs.get("timeseries", []),
            }
            return reconciled

        if m_valid:
            # Fallback: Manual
            data_cat = CAT_DEMO if is_mock else CAT_OBSERVED
            prov = PROV_DEMO_MANUAL if is_mock else PROV_OBS_MANUAL
            reason = f"Telemetry unavailable ({t_flag}); fallback to verified manual staff gauge observation ({m_flag})."
            logger.info(f"River level reconciliation fallback activated: {reason}")
            reconciled = {
                "observed_level_m": m_level,
                "timestamp": manual_obs.get("timestamp"),
                "data_category": data_cat,
                "source": "India-WRIS/CWC Manual",
                "provenance": prov,
                "unit": "m",
                "station_code": self.config.station_code,
                "station_name": self.config.station_name,
                "river": self.config.river_name,
                "danger_level_m": self.config.danger_level_m,
                "hfl_m": self.config.hfl_m,
                "retrieved_at": now.isoformat(),
                "data_age_minutes": m_age,
                "detail": reason,
                "reconciliation_method": "MANUAL_FALLBACK",
                "source_stream": "MANUAL",
                "is_fallback_active": True,
                "qc_flag": m_flag,
                "telemetry_status": telemetry_status,
                "manual_status": manual_status,
                "timeseries": manual_obs.get("timeseries", []),
            }
            return reconciled

        # 4. Neither source is valid -> UNAVAILABLE
        detail_unavail = (
            f"Observed river level unavailable. Telemetry: {t_flag}; Manual: {m_flag}. "
            "Never substituted with 0 or GloFAS modelled discharge."
        )
        return {
            "observed_level_m": None,
            "timestamp": None,
            "data_category": CAT_UNAVAILABLE,
            "source": "India-WRIS/CWC",
            "provenance": PROV_UNAVAILABLE,
            "unit": "m",
            "station_code": self.config.station_code,
            "station_name": self.config.station_name,
            "river": self.config.river_name,
            "danger_level_m": self.config.danger_level_m,
            "hfl_m": self.config.hfl_m,
            "retrieved_at": now.isoformat(),
            "data_age_minutes": None,
            "detail": detail_unavail,
            "reconciliation_method": "UNAVAILABLE",
            "source_stream": None,
            "is_fallback_active": False,
            "qc_flag": QC_MISSING,
            "telemetry_status": telemetry_status,
            "manual_status": manual_status,
            "timeseries": [],
        }

    def _regularize_series(self, series: pd.Series | None) -> pd.Series | None:
        """Snaps series timestamps to regular hourly grid, enforces _IST tz, and deduplicates."""
        if series is None or series.empty:
            return None
        s = series.copy()
        if s.index.tz is None:
            s.index = s.index.tz_localize(_IST)
        else:
            s.index = s.index.tz_convert(_IST)
        s.index = s.index.round("1h")
        s = s[~s.index.duplicated(keep="last")]
        return s.sort_index()

    def reconcile_timeseries(
        self,
        telemetry_series: pd.Series | None,
        manual_series: pd.Series | None,
        lookback_hours: int = 72,
        max_interp_gap_hours: int = 2,
        is_mock: bool = False,
    ) -> pd.DataFrame:
        """
        Reconciles two time series onto a regular hourly grid.
        - Primary source: valid Telemetry.
        - Secondary source: valid Manual.
        - Small gaps (<= max_interp_gap_hours): linear interpolation, explicitly labelled
          'DERIVED — INTERPOLATED' with is_observed=False.
        - Larger gaps: left as NaN / UNAVAILABLE.

        Returns DataFrame indexed by DatetimeIndex with columns:
        ['level_m', 'source_stream', 'data_category', 'provenance', 'is_observed', 'quality_flag']
        """
        t_reg = self._regularize_series(telemetry_series)
        m_reg = self._regularize_series(manual_series)

        now = datetime.datetime.now(_IST).replace(minute=0, second=0, microsecond=0)

        all_times = []
        if t_reg is not None and not t_reg.empty:
            all_times.extend(t_reg.index)
        if m_reg is not None and not m_reg.empty:
            all_times.extend(m_reg.index)

        if all_times:
            start_time = min(all_times)
            end_time = max(all_times)
            hourly_idx = pd.date_range(start=start_time, end=end_time, freq="1h", tz=_IST)
        else:
            start_time = now - datetime.timedelta(hours=lookback_hours)
            hourly_idx = pd.date_range(start=start_time, end=now, freq="1h", tz=_IST)

        df_out = pd.DataFrame(index=hourly_idx)
        df_out["level_m"] = np.nan
        df_out["source_stream"] = None
        df_out["data_category"] = CAT_UNAVAILABLE
        df_out["provenance"] = PROV_UNAVAILABLE
        df_out["is_observed"] = False
        df_out["quality_flag"] = QC_MISSING

        # Filter QC on both regularized input series if provided
        t_qc = self.checker.filter_series(t_reg) if t_reg is not None and not t_reg.empty else pd.DataFrame()
        m_qc = self.checker.filter_series(m_reg) if m_reg is not None and not m_reg.empty else pd.DataFrame()

        # Merge onto hourly grid
        prov_telemetry = PROV_DEMO_TELEMETRY if is_mock else PROV_OBS_TELEMETRY
        prov_manual = PROV_DEMO_MANUAL if is_mock else PROV_OBS_MANUAL
        data_cat_obs = CAT_DEMO if is_mock else CAT_OBSERVED

        for t in hourly_idx:
            # Check telemetry at this timestamp
            if not t_qc.empty and t in t_qc.index and bool(t_qc.loc[t, "is_valid"]):
                val = float(t_qc.loc[t, "level_m"])
                df_out.loc[t, "level_m"] = val
                df_out.loc[t, "source_stream"] = "TELEMETRY"
                df_out.loc[t, "data_category"] = data_cat_obs
                df_out.loc[t, "provenance"] = prov_telemetry
                df_out.loc[t, "is_observed"] = True
                df_out.loc[t, "quality_flag"] = str(t_qc.loc[t, "qc_flag"])
                continue

            # Fall back to manual at this timestamp
            if not m_qc.empty and t in m_qc.index and bool(m_qc.loc[t, "is_valid"]):
                val = float(m_qc.loc[t, "level_m"])
                df_out.loc[t, "level_m"] = val
                df_out.loc[t, "source_stream"] = "MANUAL"
                df_out.loc[t, "data_category"] = data_cat_obs
                df_out.loc[t, "provenance"] = prov_manual
                df_out.loc[t, "is_observed"] = True
                df_out.loc[t, "quality_flag"] = FLAG_FALLBACK_MANUAL
                continue

        # Handle small missing gaps with explicitly marked DERIVED interpolation
        if max_interp_gap_hours > 0:
            observed_series = df_out["level_m"].copy()
            interpolated_series = observed_series.interpolate(
                method="time", limit=max_interp_gap_hours, limit_direction="forward", limit_area="inside"
            )

            for t in hourly_idx:
                if pd.isna(df_out.loc[t, "level_m"]) and not pd.isna(interpolated_series.loc[t]):
                    interp_val = round(float(interpolated_series.loc[t]), 2)
                    df_out.loc[t, "level_m"] = interp_val
                    df_out.loc[t, "source_stream"] = "INTERPOLATED"
                    df_out.loc[t, "data_category"] = CAT_DERIVED
                    df_out.loc[t, "provenance"] = PROV_DERIVED_INTERPOLATED
                    df_out.loc[t, "is_observed"] = False  # NEVER ground truth
                    df_out.loc[t, "quality_flag"] = FLAG_INTERPOLATED

        return df_out
