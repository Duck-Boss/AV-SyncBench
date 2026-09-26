import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_models import Artifact, download


class ModelDownloadTests(unittest.TestCase):
    def test_complete_partial_is_verified_without_network_request(self) -> None:
        payload = b"complete"
        artifact = Artifact(
            model="test",
            filename="model.bin",
            url="https://example.invalid/model.bin",
            size=len(payload),
            md5=hashlib.md5(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / artifact.filename
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(payload)
            with patch(
                "scripts.download_models.urllib.request.urlopen"
            ) as urlopen:
                result = download(artifact, destination)
            urlopen.assert_not_called()
            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())


if __name__ == "__main__":
    unittest.main()
