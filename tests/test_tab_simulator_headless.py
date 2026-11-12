"""Simulator integration tests against the Streamlit app."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def simulator():
    return AppTest.from_string(
        "from app.components.tab_simulator import render\nrender()",
        default_timeout=30,
    ).run()


def test_tab_simulator_render_headless_default(simulator):
    assert not simulator.exception
    assert not simulator.error
    assert [m.label for m in simulator.metric] == ["Việt Nam", "Hoa Kỳ", "Hàn Quốc"]
    assert [m.value for m in simulator.metric] == ["Top ≈21%", "Top ≈32%", "Top ≈26%"]
    assert len(simulator.get("plotly_chart")) == 3


def test_tab_simulator_render_headless_submitted(simulator):
    simulator.slider[0].set_value(-12.0)
    simulator.button[0].click().run()
    assert not simulator.exception
    assert not simulator.error
    results = simulator.session_state["simulator_raw_predictions"]
    assert set(results) == {"VN", "US", "KR"}
    assert simulator.session_state["simulator_inputs"]["loudness_proxy"] == -12.0
    values = [m.value for m in simulator.metric]

    simulator.run()
    assert [m.value for m in simulator.metric] == values
    assert simulator.session_state["simulator_raw_predictions"] == results
