# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from collections import OrderedDict
from time import perf_counter


class PerformanceMeter:
    def __init__(self, logger):
        self.logger = logger
        self.levels = OrderedDict()
        self.level_timers = []
        self.checkpoint_timers = []

    def start_level(self, level_label, level_name, **kwargs):
        """Enters a new level in the performance meter and logs it.

        This function also creates a checkpoint on the newly created level.

        Args:
            level_label (string): The label of the new level. This could i.e. be 'task'
            level_name (string): The name of the specified level. This could i.e. be
                'tracy_todo'
            **kwargs: Any other kwargs are appended to the logging.

        Returns:
            self
        """
        self.levels[level_label] = level_name

        self.logger.info(
            f"{level_label.capitalize()} started",
            **self.levels,
            **kwargs,
        )
        time = perf_counter()
        self.level_timers.append(time)
        self.checkpoint_timers.append(time)

        return self

    def checkpoint(self, name_checkpoint, **kwargs):
        """Creates a timing checkpoint and logs the runtime from the previous one.

        Args:
            name_checkpoint (string): The name of the checkpoint. This will be logged as
                checkpoint: name_checkpoint
            **kwargs: Any other kwargs are appended to the logging.

        Returns:
            self
        """
        runtime = round(perf_counter() - self.checkpoint_timers.pop(), ndigits=3)
        self.logger.info(
            f"{name_checkpoint.capitalize()} completed",
            **self.levels,
            ktp_checkpoint=name_checkpoint,
            ktp_runtime=runtime,
            **kwargs,
        )
        self.checkpoint_timers.append(perf_counter())

        return self

    def complete_level(self, successful=True, **kwargs):
        """Completes the most inner level and logs the total runtime of that level.

        Args:
            successful (bool, optional): Whether the level was successful. Defaults to
                True.
            **kwargs: Any other kwargs are appended to the logging.

        Returns:
            self
        """

        runtime = round(perf_counter() - self.level_timers.pop(), ndigits=3)
        self.checkpoint_timers.pop()

        level_label, level_name = self.levels.popitem()

        self.logger.info(
            f"{level_label.capitalize()} completed",
            **self.levels,
            **{level_label: level_name},
            ktp_runtime=runtime,
            ktp_successful=int(successful),
            **kwargs,
        )

        return self
