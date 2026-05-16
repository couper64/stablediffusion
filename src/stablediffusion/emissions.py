"""Energy tracking: CodeCarbon for CPU/RAM, GPU estimate when NVML power sensors are absent."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, Optional

from codecarbon import EmissionsTracker

_GPU_SENSORS_AVAILABLE     : Optional[bool]              = None
_GPU_SKIP_PATCHED          : bool                        = False
_ORIGINAL_SET_GPU_TRACKING : Optional[Callable[..., None]] = None
_IDLE_POWER_FRACTION       : float                       = 0.15


@dataclass
class GpuEnergyEstimate:
    gpu_index           : int
    gpu_name            : str
    duration_seconds    : float
    power_limit_w       : float
    avg_utilization_pct : float
    estimated_power_w   : float
    energy_kwh          : float
    method              : str    = "utilization_x_power_limit"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _SuppressCodecarbonGpuLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "Failed to retrieve gpu" in message:
            return False
        if "gpu total energy consumption" in message.lower():
            return False
        return True


def reset_gpu_sensor_cache() -> None:
    """Clear cached NVML probe and restore CodeCarbon GPU hooks (for tests)."""
    global _GPU_SENSORS_AVAILABLE, _GPU_SKIP_PATCHED, _ORIGINAL_SET_GPU_TRACKING
    _GPU_SENSORS_AVAILABLE = None
    if _GPU_SKIP_PATCHED and _ORIGINAL_SET_GPU_TRACKING is not None:
        import codecarbon.core.resource_tracker as resource_tracker

        resource_tracker.ResourceTracker.set_GPU_tracking = _ORIGINAL_SET_GPU_TRACKING
        _GPU_SKIP_PATCHED = False
        _ORIGINAL_SET_GPU_TRACKING = None


def configure_codecarbon_logging() -> None:
    """Silence CodeCarbon GPU NVML warnings when we estimate GPU energy instead."""
    if gpu_energy_sensors_available():
        return
    from codecarbon.external.logger import set_logger_level

    set_logger_level("error")
    logger = logging.getLogger("codecarbon")
    logger.setLevel(logging.ERROR)
    filt = _SuppressCodecarbonGpuLogFilter()
    if not any(isinstance(f, _SuppressCodecarbonGpuLogFilter) for f in logger.filters):
        logger.addFilter(filt)
    for handler in logger.handlers:
        if not any(isinstance(f, _SuppressCodecarbonGpuLogFilter) for f in handler.filters):
            handler.addFilter(filt)


def _install_codecarbon_gpu_skip() -> None:
    """Prevent CodeCarbon from polling NVML GPU power APIs (still estimates GPU ourselves)."""
    global _GPU_SKIP_PATCHED, _ORIGINAL_SET_GPU_TRACKING
    if gpu_energy_sensors_available() or _GPU_SKIP_PATCHED:
        return

    import codecarbon.core.resource_tracker as resource_tracker

    _ORIGINAL_SET_GPU_TRACKING = resource_tracker.ResourceTracker.set_GPU_tracking

    def _skip_gpu_tracking(self) -> None:
        self.gpu_tracker = "disabled"
        self.tracker._conf["gpu_count"] = 0
        self.tracker._conf.setdefault("gpu_model", "")

    resource_tracker.ResourceTracker.set_GPU_tracking = _skip_gpu_tracking
    _GPU_SKIP_PATCHED = True


def _apply_no_sensor_workarounds() -> None:
    configure_codecarbon_logging()
    _install_codecarbon_gpu_skip()


def gpu_energy_sensors_available() -> bool:
    """Return True when NVML exposes GPU power or energy counters."""
    global _GPU_SENSORS_AVAILABLE
    if _GPU_SENSORS_AVAILABLE is not None:
        return _GPU_SENSORS_AVAILABLE

    available = False
    try:
        import pynvml

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            _GPU_SENSORS_AVAILABLE = False
            _apply_no_sensor_workarounds()
            return False
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        for probe in (
            pynvml.nvmlDeviceGetPowerUsage,
            pynvml.nvmlDeviceGetTotalEnergyConsumption,
        ):
            try:
                probe(handle)
                available = True
                break
            except pynvml.NVMLError:
                continue
    except Exception:
        available = False

    _GPU_SENSORS_AVAILABLE = available
    if not available:
        _apply_no_sensor_workarounds()
    return available


class GpuPowerEstimator:
    """Estimate GPU energy from utilization and enforced power limit."""

    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = gpu_index
        self._handle: Any = None
        self._name = ""
        self._power_limit_w = 0.0
        self._samples: list[float] = []

    def init(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self._name = name.decode() if isinstance(name, bytes) else str(name)
            limit_mw = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle)
            self._power_limit_w = limit_mw / 1000.0
            return True
        except Exception:
            return False

    def sample(self) -> None:
        if self._handle is None:
            return
        import pynvml

        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        self._samples.append(float(util))

    def estimate(self, duration_seconds: float) -> Optional[GpuEnergyEstimate]:
        if self._handle is None or duration_seconds <= 0:
            return None
        if not self._samples:
            self.sample()
        avg_util = sum(self._samples) / len(self._samples) if self._samples else 0.0
        load = _IDLE_POWER_FRACTION + (1.0 - _IDLE_POWER_FRACTION) * (avg_util / 100.0)
        estimated_power_w = self._power_limit_w * load
        energy_kwh = estimated_power_w * duration_seconds / 3_600_000.0
        return GpuEnergyEstimate(
            gpu_index           = self.gpu_index,
            gpu_name            = self._name,
            duration_seconds    = duration_seconds,
            power_limit_w       = self._power_limit_w,
            avg_utilization_pct = avg_util,
            estimated_power_w   = estimated_power_w,
            energy_kwh          = energy_kwh,
        )


@contextmanager
def gpu_utilization_sampler(
    gpu_index: int = 0,
    interval: float = 0.5,
) -> Iterator[Optional[GpuPowerEstimator]]:
    """Sample GPU utilization while the caller runs; no-op when NVML power sensors work."""
    if gpu_energy_sensors_available():
        yield None
        return

    estimator = GpuPowerEstimator(gpu_index)
    if not estimator.init():
        yield None
        return

    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            estimator.sample()
            stop.wait(interval)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    try:
        yield estimator
    finally:
        stop.set()
        thread.join(timeout=interval + 1.0)


def create_emissions_tracker(project_name: str) -> EmissionsTracker:
    """Build a CodeCarbon tracker; skip GPU NVML polling when sensors are unavailable."""
    use_sensors = gpu_energy_sensors_available()
    kwargs: Dict[str, Any] = {
        "project_name" : project_name,
        "save_to_file" : False,
        "log_level"    : "error",
    }
    if not use_sensors:
        kwargs["gpu_ids"] = []
        print(
            "GPU energy: NVML power sensors unavailable; "
            "using utilization × power-limit estimate for GPU."
        )
    return EmissionsTracker(**kwargs)


@contextmanager
def track_emissions(project_name: str) -> Iterator[Dict[str, Any]]:
    """Track energy for a code block using CodeCarbon's context-manager pattern.

    Example::

        with track_emissions("my-project") as energy:
            run_training()
        print(energy["energy_kwh"], energy["emissions_kg_co2"])
    """
    results: Dict[str, Any] = {}
    seconds = 0.0
    gpu_estimate: Optional[GpuEnergyEstimate] = None
    tracker = create_emissions_tracker(project_name)
    with tracker:
        with gpu_utilization_sampler() as gpu_estimator:
            start = time.perf_counter()
            yield results
            seconds = time.perf_counter() - start
            if gpu_estimator is not None:
                gpu_estimate = gpu_estimator.estimate(seconds)
    emissions = tracker_emissions_dict(tracker)
    emissions_kg_co2: Optional[float] = None
    if tracker.final_emissions_data is not None:
        emissions_kg_co2 = tracker.final_emissions_data.emissions
    results.update(
        combine_energy_metrics(emissions, emissions_kg_co2, seconds, gpu_estimate)
    )
    results["seconds"] = seconds
    results["codecarbon"] = emissions


def tracker_emissions_dict(tracker: EmissionsTracker) -> Dict[str, Any]:
    if tracker.final_emissions_data is None:
        return {}
    return asdict(tracker.final_emissions_data)


def combine_energy_metrics(
    emissions: Dict[str, Any],
    emissions_kg_co2: Optional[float],
    duration_seconds: float,
    gpu_estimate: Optional[GpuEnergyEstimate],
) -> Dict[str, Any]:
    """Merge CodeCarbon CPU/RAM totals with optional estimated GPU energy."""
    use_sensors = gpu_energy_sensors_available()
    codecarbon_kwh = float(emissions.get("energy_consumed") or 0.0)
    codecarbon_emissions = float(emissions_kg_co2 or 0.0)

    if use_sensors:
        return {
            "energy_kwh"       : codecarbon_kwh,
            "emissions_kg_co2" : codecarbon_emissions,
            "energy_breakdown" : {
                "total_kwh"  : codecarbon_kwh,
                "gpu_kwh"    : float(emissions.get("gpu_energy") or 0.0),
                "gpu_method" : "nvml",
            },
        }

    gpu_kwh = gpu_estimate.energy_kwh if gpu_estimate else 0.0
    total_kwh = codecarbon_kwh + gpu_kwh
    if codecarbon_kwh > 0 and gpu_kwh > 0:
        total_emissions = codecarbon_emissions * (total_kwh / codecarbon_kwh)
    elif gpu_kwh > 0:
        total_emissions = codecarbon_emissions + gpu_kwh * 0.475
    else:
        total_emissions = codecarbon_emissions

    breakdown: Dict[str, Any] = {
        "total_kwh"        : total_kwh,
        "cpu_ram_kwh"      : codecarbon_kwh,
        "gpu_kwh"          : gpu_kwh,
        "gpu_method"       : "utilization_x_power_limit" if gpu_estimate else None,
        "duration_seconds" : duration_seconds,
    }
    if gpu_estimate is not None:
        breakdown["gpu"] = gpu_estimate.as_dict()

    return {
        "energy_kwh"       : total_kwh,
        "emissions_kg_co2" : total_emissions,
        "energy_breakdown" : breakdown,
    }
