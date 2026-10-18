"""Browser test: the web app's in-browser predictions must match the Python ONNX predictor.

Needs Playwright with Chromium (pip install playwright && playwright install chromium); skipped otherwise.
Set EDN_CHROMIUM to an existing Chromium/Chrome/Edge binary to skip the Playwright download.
Set EDN_WEB_ASSETS to a folder containing three.min.js and onnxruntime-web's dist/ files to run offline.
"""

import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from edgedefectnet.predict import Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def site_url(tmp_path_factory):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    site = mod.build(tmp_path_factory.mktemp("site") / "_site")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def _offline_routes(ctx):
    assets = os.environ.get("EDN_WEB_ASSETS")
    if not assets:
        return
    ctx.route("**/three.min.js", lambda r: r.fulfill(path=os.path.join(assets, "three.min.js")))

    def ort(r):
        name = r.request.url.split("/dist/")[-1].split("?")[0]
        ctype = "application/wasm" if name.endswith(".wasm") else "application/javascript"
        r.fulfill(path=os.path.join(assets, "dist", name), content_type=ctype)

    ctx.route("**/onnxruntime-web@*/dist/**", ort)


@pytest.fixture(scope="module")
def browser():
    executable = os.environ.get("EDN_CHROMIUM")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"Chromium not available: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser, site_url):
    ctx = browser.new_context()
    _offline_routes(ctx)
    pg = ctx.new_page()
    pg.errors = []  # type: ignore[attr-defined]
    pg.on("pageerror", lambda exc: pg.errors.append(str(exc)))  # type: ignore[attr-defined]
    pg.goto(site_url, wait_until="domcontentloaded")
    pg.locator(".example").first.wait_for()
    yield pg
    ctx.close()


def test_browser_predictions_match_python(page):
    predictor = Predictor(ROOT / "models" / "edgedefectnet.onnx")
    expected = {p.name: predictor.predict(p).label for p in sorted((ROOT / "examples").glob("*.png"))}
    buttons = page.locator(".example")
    got = {}
    for i in range(buttons.count()):
        page.evaluate("delete document.body.dataset.predicted")
        buttons.nth(i).click()
        page.wait_for_selector("body[data-predicted]", timeout=120_000)
        name = buttons.nth(i).locator("img").get_attribute("src").split("/")[-1]
        got[name] = page.evaluate("document.body.dataset.predicted")
    assert page.locator(".matrix .cell").count() == 81
    assert page.locator(".class-card").count() == 9
    assert page.locator("#legend button").count() == 9
    assert got == expected
    assert page.errors == []


def test_uploaded_file_is_classified_like_python(page):
    example = sorted((ROOT / "examples").glob("Cracks__*.png"))[0]
    expected = Predictor(ROOT / "models" / "edgedefectnet.onnx").predict(example).label
    page.set_input_files("#file", str(example))
    page.wait_for_selector("body[data-predicted]", timeout=120_000)
    assert page.evaluate("document.body.dataset.predicted") == expected
    assert page.locator("#result").get_attribute("data-state") == "done"
    assert page.locator("#bars li").count() == 9


def test_non_image_upload_shows_an_error(page, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not an image")
    page.set_input_files("#file", str(bad))
    page.wait_for_selector('#result[data-state="error"]')
    assert "isn't an image" in page.locator("#errorText").inner_text()


def test_theme_toggle_switches_and_persists(page):
    start = page.evaluate("document.documentElement.dataset.theme")
    page.click("#theme")
    flipped = page.evaluate("document.documentElement.dataset.theme")
    assert flipped != start
    assert page.evaluate("localStorage.getItem('edn-theme')") == flipped
    page.reload()
    assert page.evaluate("document.documentElement.dataset.theme") == flipped
