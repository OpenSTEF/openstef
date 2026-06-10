# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""macOS 26 gunicorn worker crash workaround.

On macOS 26 (Tahoe / Darwin 25), setproctitle's darwin_set_process_title calls
CFBundleGetFunctionPointerForName in a forked child process.  That call goes through
_os_log_preferences_refresh, which is not fork-safe in macOS 26 and raises SIGSEGV.

Airflow's hardcoded gunicorn config (airflow.www.gunicorn_config) calls
setproctitle.setproctitle() in its post_worker_init hook, triggered for every worker
immediately after fork.  Replacing the function with a no-op prevents the crash.
Process titles are cosmetic only — no functionality is affected.

This file is loaded automatically as sitecustomize.py when PYTHONPATH includes this
directory, which the deploy-airflow-ui poe task arranges.  It has no effect on Linux.
"""

import platform as _platform

if _platform.system() == "Darwin":
    try:
        import setproctitle as _spt

        _spt.setproctitle = lambda title: None  # type: ignore[assignment]
        _spt.getproctitle = lambda: ""  # type: ignore[assignment]
    except ImportError:
        pass
