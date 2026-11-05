"""Common fit/predict interface and JSON save/load of parameters."""

import json
from pathlib import Path


class Model:
    name = "model"

    def fit(self, t, y):
        raise NotImplementedError

    def predict(self, t):
        raise NotImplementedError

    def params(self) -> dict:
        """Fitted parameters as plain JSON-serialisable values."""
        raise NotImplementedError

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.name, **self.params()}, indent=2))
