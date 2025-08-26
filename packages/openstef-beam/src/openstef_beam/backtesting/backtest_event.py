# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Core event types that drive backtesting simulations.

Events represent discrete training and prediction operations that occur
during backtesting, capturing when each operation should happen and
enabling batch processing for efficiency.
"""

from datetime import datetime
from typing import Literal

from openstef_core.base_model import BaseModel


class BacktestEvent(BaseModel):
    """Represents a single event during a backtest simulation.

    An event can be either a prediction or training operation that occurs at a specific timestamp.
    """

    type: Literal["predict", "train"]
    timestamp: datetime


class BacktestEventBatch(BaseModel):
    """Represents a batch of backtest events that can be processed together.

    Encapsulates the logic for determining if events can be batched and
    provides a clean interface for processing different types of event groups.
    """

    events: list[BacktestEvent]

    @property
    def batch_type(self) -> Literal["train", "predict_batch", "predict_single"]:
        """Determine the type of batch for processing logic."""
        if len(self.events) == 1:
            return "train" if self.events[0].type == "train" else "predict_single"
        else:  # noqa: RET505 - kept for readability
            return "predict_batch"

    @property
    def is_training(self) -> bool:
        """Check if this batch contains a training event."""
        return self.batch_type == "train"

    @property
    def is_prediction(self) -> bool:
        """Check if this batch contains prediction events."""
        return self.batch_type in {"predict_batch", "predict_single"}

    def __len__(self) -> int:
        """Return the number of events in this batch."""
        return len(self.events)
