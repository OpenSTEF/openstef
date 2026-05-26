# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Integration components for extending OpenSTEF functionality.

Contains implementations for callbacks and storage systems that hook into and
extend OpenSTEF functionality by integrating with external systems such as
monitoring tools, databases, cloud storage, and custom processing pipelines.
"""

__all__ = ["joblib", "mlflow", "optuna"]  # noqa: F822  # pyright: ignore[reportUnsupportedDunderAll]  # Sub-packages with optional deps; not imported to avoid missing-extra errors at import time
