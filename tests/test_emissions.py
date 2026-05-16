from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stablediffusion.emissions import (
    GpuPowerEstimator,
    combine_energy_metrics,
    create_emissions_tracker,
    gpu_energy_sensors_available,
    reset_gpu_sensor_cache,
    track_emissions,
)


@pytest.fixture(autouse=True)
def _clear_gpu_sensor_cache() -> None:
    reset_gpu_sensor_cache()
    yield
    reset_gpu_sensor_cache()


def test_gpu_energy_sensors_available_when_power_probe_works() -> None:
    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_nvml.NVMLError = type("NVMLError", (Exception,), {})

    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        assert gpu_energy_sensors_available() is True


def test_gpu_energy_sensors_unavailable_when_both_probes_fail() -> None:
    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_nvml.NVMLError = type("NVMLError", (Exception,), {})
    mock_nvml.nvmlDeviceGetPowerUsage.side_effect = mock_nvml.NVMLError()
    mock_nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = mock_nvml.NVMLError()

    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        assert gpu_energy_sensors_available() is False


def test_gpu_power_estimator_estimate() -> None:
    est = GpuPowerEstimator(0)
    est._handle = object()
    est._name = "NVIDIA GeForce RTX 4060"
    est._power_limit_w = 115.0
    est._samples = [80.0, 90.0]

    result = est.estimate(10.0)
    assert result is not None
    assert result.avg_utilization_pct == 85.0
    assert result.estimated_power_w == pytest.approx(115.0 * (0.15 + 0.85 * 0.85))
    assert result.energy_kwh == pytest.approx(result.estimated_power_w * 10.0 / 3_600_000.0)


def test_track_emissions_context_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTracker:
        final_emissions_data = None

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "stablediffusion.emissions.create_emissions_tracker",
        lambda _name: _FakeTracker(),
    )
    monkeypatch.setattr(
        "stablediffusion.emissions.gpu_utilization_sampler",
        lambda *args, **kwargs: __import__("contextlib").nullcontext(None),
    )
    monkeypatch.setattr(
        "stablediffusion.emissions.tracker_emissions_dict",
        lambda _tracker: {"energy_consumed": 0.001},
    )
    monkeypatch.setattr(
        "stablediffusion.emissions.gpu_energy_sensors_available",
        lambda: True,
    )

    with track_emissions("test") as energy:
        pass

    assert energy["seconds"] >= 0
    assert energy["emissions_kg_co2"] == 0.0
    assert energy["energy_kwh"] == 0.001


def test_create_emissions_tracker_skips_gpu_nvml_when_sensors_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stablediffusion.emissions.gpu_energy_sensors_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "stablediffusion.emissions.EmissionsTracker",
        lambda **kwargs: kwargs,
    )
    kwargs = create_emissions_tracker("test-project")
    assert kwargs["gpu_ids"] == []
    assert kwargs["log_level"] == "error"


def test_combine_energy_metrics_adds_gpu_estimate() -> None:
    gpu = GpuPowerEstimator(0)
    gpu._handle = object()
    gpu._name = "GPU"
    gpu._power_limit_w = 100.0
    gpu._samples = [100.0]
    estimate = gpu.estimate(3600.0)
    assert estimate is not None

    with patch(
        "stablediffusion.emissions.gpu_energy_sensors_available",
        return_value=False,
    ):
        merged = combine_energy_metrics(
            {"energy_consumed": 0.001},
            0.0005,
            3600.0,
            estimate,
        )

    assert merged["energy_kwh"] == pytest.approx(0.001 + estimate.energy_kwh)
    assert merged["energy_breakdown"]["gpu_method"] == "utilization_x_power_limit"
