import json

import pytest

import voicelab.rvc_assets as assets


def test_upload_wait_has_a_deadline_and_never_downloads_on_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    (tmp_path / "ASSETS_UPLOAD_PENDING").write_text("pending")
    clock = iter((0.0, 901.0))
    monkeypatch.setattr(assets.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Must wait for upload"))
    with pytest.raises(TimeoutError, match="15 minutes"):
        assets.download(tmp_path / "assets.json")
    assert not (tmp_path / "assets.json").exists()


def test_upload_release_resumes_after_marker_is_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "ASSETS", {})
    marker = tmp_path / "ASSETS_UPLOAD_PENDING"
    marker.write_text("pending")
    monkeypatch.setattr(assets.time, "sleep", lambda _: marker.unlink())
    assert assets.download(tmp_path / "assets.json") == []
    assert json.loads((tmp_path / "assets.json").read_text())["revision"] == assets.REVISION
