"""Verify retained artifact bytes; pass the local download directory as argv[1]."""
from pathlib import Path
import email
import hashlib
import json
import sys
import tarfile
import zipfile

evidence = Path(__file__).parent
root = Path(sys.argv[1])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


downloads = json.loads((evidence / "download-verification.json").read_text())
assert len(downloads) == 24
for artifact in downloads:
    assert digest(root / "artifacts" / (artifact["name"] + ".zip")) == artifact["zip_sha256"]
    for file in artifact["files"]:
        path = root / file["path"]
        assert path.stat().st_size == file["size"]
        assert digest(path) == file["sha256"], path
for name in ("npm-artifact", "fixed-npm-artifact"):
    metadata = json.loads((evidence / (name + "-metadata.json")).read_text())
    assert "sha256:" + digest(root / (name + ".zip")) == metadata["digest"]
for directory in (root / "artifacts/vanedb-crate", root / "npm-artifact",
                  root / "fixed-npm-artifact", root / "artifacts/capi-signed-release"):
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        assert digest(directory / name.lstrip("* ")) == sha, name
wheels = list((root / "artifacts").glob("vanedb-*/*.whl"))
assert len(wheels) == 28
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(names) == 1
        metadata = email.message_from_bytes(archive.read(names[0]))
        assert metadata["Name"] == "vanedb" and metadata["Version"] == "0.2.0rc2"
sdists = list((root / "artifacts/vanedb-sdist").glob("*.tar.gz"))
assert len(sdists) == 1
with tarfile.open(sdists[0]) as archive:
    metadata = email.message_from_bytes(archive.extractfile("vanedb-0.2.0rc2/PKG-INFO").read())
    assert metadata["Version"] == "0.2.0rc2"
print("PASS: 26 downloaded ZIP digests, retained file hashes, package checksum manifests, 28 wheel versions and one sdist")
