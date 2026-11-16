import shutil
from pathlib import Path

import pytest
import yaml

from wct2g1.config import DEFAULT_CONFIG, load_config


@pytest.fixture
def cfg(tmp_path: Path):
    """The real config, with data paths redirected into a temp dir."""
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return load_config(p)


requires_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)
