"""
loader.py — scan a projections directory and serve DataFrames on demand.

Directory layout expected (produced by reduce_activations.py / ProjectionSuite):

  <root>/
    layer_00_umap.parquet   (or .csv)
    layer_00_pca.parquet
    layer_06_umap.parquet
    ...
    suite_summary.csv         ← PCA explained-variance table (optional)

ProjectionStore is the main class.  It caches loaded DataFrames with
@st.cache_data so layer switches are instant after the first load.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^layer_(\d+)_(umap|pca)\.(parquet|csv)$", re.I)


# ---------------------------------------------------------------------------
# Entry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionEntry:
    path:   Path
    layer:  int
    method: str   # "umap" | "pca"
    fmt:    str   # "parquet" | "csv"


# ---------------------------------------------------------------------------
# Cached loader (module-level so Streamlit can hash it across reruns)
# ---------------------------------------------------------------------------

def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# ProjectionStore
# ---------------------------------------------------------------------------

class ProjectionStore:
    """
    Scan a directory for saved projection files and provide fast access.

    Parameters
    ----------
    root : str | Path
        Directory created by ``ProjectionSuite.save()`` or
        ``reduce_activations.py``.

    Usage
    -----
    >>> store = ProjectionStore("projections/run_01")
    >>> store.available_layers("umap")
    [0, 3, 6, 9, 11]
    >>> df = store.load(6, "umap")
    >>> df.columns
    Index(['x', 'y', 'label', 'layer', 'text', 'activation_norm', ...])
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._entries: List[ProjectionEntry] = []
        self._df_cache: Dict[Tuple[int, str], pd.DataFrame] = {}
        self._summary: Optional[pd.DataFrame] = None
        self._scan()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def available_methods(self) -> List[str]:
        """Sorted list of methods found in the directory (e.g. ["pca", "umap"])."""
        return sorted({e.method for e in self._entries})

    def available_layers(self, method: str) -> List[int]:
        """Sorted list of layer indices available for *method*."""
        return sorted({e.layer for e in self._entries if e.method == method})

    def has(self, layer: int, method: str) -> bool:
        return any(e.layer == layer and e.method == method for e in self._entries)

    @property
    def summary(self) -> Optional[pd.DataFrame]:
        """PCA explained-variance summary (None if not present)."""
        return self._summary

    @property
    def all_entries(self) -> List[ProjectionEntry]:
        return list(self._entries)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, layer: int, method: str) -> Optional[pd.DataFrame]:
        """
        Load and cache the DataFrame for (*layer*, *method*).

        Returns None if the file is not found.
        """
        key = (layer, method)
        if key in self._df_cache:
            return self._df_cache[key]

        entry = self._entry(layer, method)
        if entry is None:
            logger.warning("No projection file for layer=%d method=%s", layer, method)
            return None

        df = self._read(entry)
        self._df_cache[key] = df
        logger.debug("Loaded %s  (%d rows × %d cols)", entry.path.name, len(df), len(df.columns))
        return df

    def load_multi(
        self,
        layers: List[int],
        method: str,
    ) -> Optional[pd.DataFrame]:
        """
        Load and concatenate DataFrames for multiple layers.

        The "layer" column in each DataFrame is preserved so the combined
        DataFrame can be colored by layer index.
        """
        parts = []
        for layer in sorted(layers):
            df = self.load(layer, method)
            if df is not None:
                parts.append(df)
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    def clear_cache(self) -> None:
        """Evict all cached DataFrames (e.g. after the source directory changes)."""
        self._df_cache.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        if not self.root.exists():
            logger.warning("Projection directory not found: %s", self.root)
            return

        for path in sorted(self.root.iterdir()):
            m = _FILENAME_RE.match(path.name)
            if m:
                layer  = int(m.group(1))
                method = m.group(2).lower()
                fmt    = m.group(3).lower()
                self._entries.append(ProjectionEntry(path=path, layer=layer,
                                                     method=method, fmt=fmt))

        summary_path = self.root / "suite_summary.csv"
        if summary_path.exists():
            self._summary = pd.read_csv(summary_path)

        logger.info(
            "ProjectionStore: found %d files in %s  (methods=%s)",
            len(self._entries), self.root, self.available_methods(),
        )

    def _entry(self, layer: int, method: str) -> Optional[ProjectionEntry]:
        for e in self._entries:
            if e.layer == layer and e.method == method:
                return e
        return None

    def _read(self, entry: ProjectionEntry) -> pd.DataFrame:
        path_str = str(entry.path)
        if entry.fmt == "parquet":
            return _load_parquet(path_str)
        return _load_csv(path_str)

    def __repr__(self) -> str:
        return (
            f"ProjectionStore(root={self.root!r}, "
            f"entries={len(self._entries)}, "
            f"methods={self.available_methods()})"
        )
