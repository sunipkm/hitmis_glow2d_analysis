# %%
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys
from typing import List, Optional

# %%
ROOT_DIR = Path(__file__).resolve().parent


def is_interactive_session() -> bool:
    """Check if the script is running in an interactive environment."""
    return hasattr(sys, 'ps1') or hasattr(sys, 'ps2')


class Directories:
    """Class to manage directories for storing model data, and various derivatives.
    """
    def __init__(self, suffix: Optional[str] = None):
        """Initialize the Directories class with an optional suffix.

        Args:
            suffix (Optional[str], optional): Suffix to append to directory names. Defaults to None.
        """
        self._suffix = suffix.strip() if suffix else None

    def _create_path(self, name: str) -> Path:
        path = ROOT_DIR / Path('_'.join(list(filter(None, [name, self._suffix]))))
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def suffix(self) -> Optional[str]:
        """ Get the suffix used for directory names.

        Returns:
            Optional[str]: Suffix used for directory names.
        """
        return self._suffix

    @property
    def model_dir(self) -> Path:
        """Get the directory for storing model data.
        Defaults to 'keomodel' if no suffix is provided.

        Returns:
            Path: Directory for storing model data.
        """
        return self._create_path('keomodel')

    @property
    def fitprops_dir(self) -> Path:
        """Get the directory for storing fit properties.
        Defaults to 'fitprops' if no suffix is provided.

        Returns:
            Path: Directory for storing fit properties.
        """
        return self._create_path('fitprops')

    @property
    def vertprops_dir(self) -> Path:
        """Get the directory for storing vertical properties.
        Defaults to 'fitpropsvert' if no suffix is provided.

        Returns:
            Path: Directory for storing vertical properties.
        """
        return self._create_path('fitpropsvert')

    @property
    def keograms_dir(self) -> Path:
        """Get the directory for storing keograms.
        Defaults to 'keograms' if no suffix is provided.

        Returns:
            Path: Directory for storing keograms.
        """
        return self._create_path('keograms')

    @property
    def counts_dir(self) -> Path:
        """Get the directory for storing counts.
        Defaults to 'keocounts' if no suffix is provided.

        Returns:
            Path: Directory for storing counts.
        """
        return ROOT_DIR / Path('keocounts')

# %%
if __name__ == '__main__':
    # Example usage
    dirs = Directories(suffix='example')
    print(f"Model Directory: {dirs.model_dir}")
    print(f"Fit Properties Directory: {dirs.fitprops_dir}")
    print(f"Vertical Properties Directory: {dirs.vertprops_dir}")
    print(f"Keograms Directory: {dirs.keograms_dir}")
    print(f"Counts Directory: {dirs.counts_dir}")
    
    if is_interactive_session():
        print("Running in interactive mode.")
    else:
        print("Running in non-interactive mode.")
# %%
