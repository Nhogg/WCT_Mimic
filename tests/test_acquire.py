import json
import subprocess

from conftest import requires_ffmpeg
from wct2g1 import acquire
from wct2g1.manifest import read_manifest


def _fake_video(cfg, vid, fps=50, with_media=True):
    d = cfg.path_of("raw") / vid
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{vid}.info.json").write_text(json.dumps(
        {"id": vid, "title": f"t {vid}", "upload_date": "20240101", "duration": 1, "fps": fps}))
    if with_media:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x180:rate={fps}",
             "-t", "1", "-pix_fmt", "yuv420p", str(d / f"{vid}.mp4")], check=True)


def test_ytdlp_command_uses_config(cfg):
    cfg["acquire"]["match_filters"] = ["upload_date>=20240101"]
    cmd = acquire.ytdlp_command(cfg, "https://example/playlist", limit=3)
    assert cmd[0] == cfg["acquire"]["ytdlp_bin"]
    assert cmd[cmd.index("-f") + 1] == cfg["acquire"]["format"]
    assert "--write-info-json" in cmd and "--no-write-playlist-metafiles" in cmd
    assert cmd[cmd.index("--download-archive") + 1].endswith("raw/archive.txt")
    assert cmd[cmd.index("--match-filters") + 1] == "upload_date>=20240101"
    assert cmd[cmd.index("--max-downloads") + 1] == "3"
    assert cmd[-1] == "https://example/playlist"


@requires_ffmpeg
def test_manifest_matches_disk_and_prunes_archive(cfg):
    _fake_video(cfg, "vidA", fps=50)
    _fake_video(cfg, "vidB", with_media=False)
    raw = cfg.path_of("raw")
    (raw / "archive.txt").write_text("youtube vidA\nyoutube vidB\nyoutube gone\n")

    dropped = acquire.prune_archive(cfg)
    assert sorted(dropped) == ["gone", "vidB"]
    assert (raw / "archive.txt").read_text() == "youtube vidA\n"

    acquire.run(cfg, manifest_only=True)
    df = read_manifest(cfg, "raw").set_index("video_id")
    assert df.loc["vidA", "status"] == "ok"
    assert df.loc["vidA", "fps"] == 50.0 and df.loc["vidA", "height"] == 180
    assert df.loc["vidA", "path"].endswith("vidA/vidA.mp4")
    assert df.loc["vidB", "status"].startswith("failed:")
