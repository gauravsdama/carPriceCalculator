from __future__ import annotations

import json
import shutil
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "docs" / "demo"

pytestmark = pytest.mark.browser


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@contextmanager
def static_server(root: Path):
    handler = partial(QuietStaticHandler, directory=root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def browser():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1280,900")
    driver = webdriver.Chrome(options=options)
    try:
        yield driver
    finally:
        driver.quit()


def expected_estimate(artifact: dict, *, model: str, year: int, mileage: int, rating: float) -> str:
    model_index = artifact["model"]["models"].index(model)
    numeric_values = (year, mileage, rating)
    numeric_contribution = sum(
        (
            (value - artifact["model"]["numeric_mean"][index])
            / artifact["model"]["numeric_scale"][index]
        )
        * artifact["model"]["numeric_coefficients"][index]
        for index, value in enumerate(numeric_values)
    )
    value = max(
        0,
        artifact["model"]["intercept"]
        + numeric_contribution
        + artifact["model"]["model_coefficients"][model_index],
    )
    return f"${value:,.0f}"


def wait_for_status(browser, text: str):
    WebDriverWait(browser, 10).until(
        lambda driver: text in driver.find_element(By.ID, "status").text
    )


def test_browser_loads_contract_and_calculates_an_estimate(browser):
    artifact = json.loads((DEMO_ROOT / "model.json").read_text())
    with static_server(DEMO_ROOT) as url:
        browser.get(url)
        wait_for_status(browser, "Estimate updated.")

        assert browser.find_element(By.ID, "estimate").text == expected_estimate(
            artifact,
            model=artifact["defaults"]["model"],
            year=artifact["defaults"]["year"],
            mileage=artifact["defaults"]["mileage"],
            rating=artifact["defaults"]["rating"],
        )

        Select(browser.find_element(By.ID, "model")).select_by_value("C-Class C 300")
        for field, value in (("year", "2021"), ("mileage", "48000"), ("rating", "4.5")):
            element = browser.find_element(By.ID, field)
            element.clear()
            element.send_keys(value)
        browser.find_element(By.ID, "estimate-button").click()

        assert browser.find_element(By.ID, "estimate").text == expected_estimate(
            artifact,
            model="C-Class C 300",
            year=2021,
            mileage=48000,
            rating=4.5,
        )
        assert (
            browser.find_element(By.ID, "status").text == "Estimate updated."
        )


def test_browser_rejects_an_incompatible_artifact(browser, tmp_path):
    demo_copy = tmp_path / "demo"
    shutil.copytree(DEMO_ROOT, demo_copy)
    artifact_path = demo_copy / "model.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["schema_version"] = 999
    artifact_path.write_text(json.dumps(artifact))

    with static_server(demo_copy) as url:
        browser.get(url)
        wait_for_status(browser, "unsupported schema version")

        assert not browser.find_element(By.ID, "model").is_enabled()
        assert not browser.find_element(By.ID, "estimate-button").is_enabled()
        assert browser.find_element(By.ID, "estimate").text == "—"
