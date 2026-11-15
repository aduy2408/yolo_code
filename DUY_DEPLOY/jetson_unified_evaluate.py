from __future__ import annotations
import sys
import types

_pi_heif = types.ModuleType("pi_heif")
_pi_heif.register_heif_opener = lambda: None
sys.modules.setdefault("pi_heif", _pi_heif)

import ultralytics.data.utils as _data_utils
import ultralytics.utils.checks as _checks


def _skip_font_download(*args, **kwargs):
    return None


_checks.check_font = _skip_font_download
_data_utils.check_font = _skip_font_download

from evaluate import main


if __name__ == "__main__":
    main()
