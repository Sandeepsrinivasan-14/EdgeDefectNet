"""Browser test: the web app's in-browser predictions must match the Python ONNX predictor.

Needs Playwright with Chromium (pip install playwright && playwright install chromium); skipped otherwise.
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


def test_browser_predictions_match_python(site_url):
    predictor = Predictor(ROOT / "models" / "edgedefectnet.onnx")
    expected = {p.name: predictor.predict(p).label for p in sorted((ROOT / "examples").glob("*.png"))}
    executable = os.environ.get("EDN_CHROMIUM")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=executable) if executable else p.chromium.launch()
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"Chromium not available: {exc}")
        ctx = browser.new_context()
        _offline_routes(ctx)
        page = ctx.new_page()
        page.goto(site_url)
        buttons = page.locator(".example")
        buttons.first.wait_for()
        got = {}
        for i in range(buttons.count()):
            page.evaluate("delete document.body.dataset.predicted")
            buttons.nth(i).click()
            page.wait_for_selector("body[data-predicted]", timeout=120_000)
            name = buttons.nth(i).locator("img").get_attribute("src").split("/")[-1]
            got[name] = page.evaluate("document.body.dataset.predicted")
        assert page.locator(".matrix .cell").count() == 81
        browser.close()
    assert got == expected
