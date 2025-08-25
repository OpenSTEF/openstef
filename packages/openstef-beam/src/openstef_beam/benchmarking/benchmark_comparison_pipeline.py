# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from collections import defaultdict

from openstef_beam.analysis import AnalysisConfig, AnalysisPipeline, AnalysisScope
from openstef_beam.analysis.models import AnalysisAggregation, GroupName, RunName, TargetMetadata, TargetName
from openstef_beam.benchmarking.benchmark_pipeline import read_evaluation_reports
from openstef_beam.benchmarking.models import BenchmarkTarget
from openstef_beam.benchmarking.storage import BenchmarkStorage
from openstef_beam.benchmarking.target_provider import TargetProvider
from openstef_beam.evaluation import EvaluationReport

_logger = logging.getLogger(__name__)


class BenchmarkComparisonPipeline[T: BenchmarkTarget, F]:
    def __init__(
        self,
        analysis_config: AnalysisConfig,
        target_provider: TargetProvider[T, F],
        storage: BenchmarkStorage,
    ):
        super().__init__()
        self.analysis_config = analysis_config
        self.target_provider = target_provider
        self.storage = storage
        self.pipeline = AnalysisPipeline(
            config=self.analysis_config,
        )

    def run(
        self,
        run_data: dict[RunName, BenchmarkStorage],
        filter_args: F | None = None,
    ):
        targets = self.target_provider.get_targets(filter_args)

        # Read evaluation reports for each run
        reports: list[tuple[TargetMetadata, EvaluationReport]] = []
        for run_name, run_storage in run_data.items():
            run_reports = read_evaluation_reports(
                targets=targets,
                storage=run_storage,
                run_name=run_name,
                strict=True,
            )
            reports.extend(run_reports)

        self.run_global(reports)
        self.run_for_groups(reports)
        self.run_for_targets(reports)

    def run_global(self, reports: list[tuple[TargetMetadata, EvaluationReport]]):
        scope = AnalysisScope(
            aggregation=AnalysisAggregation.RUN_AND_GROUP,
        )
        if self.storage.has_analysis_output(scope=scope):
            _logger.info("Skipping global analysis, already exists")
            return

        _logger.info("Running analysis comparison for runs across groups")
        analysis = self.pipeline.run_for_reports(
            reports=reports,
            scope=scope,
        )
        self.storage.save_analysis_output(output=analysis)

    def run_for_groups(
        self,
        reports: list[tuple[TargetMetadata, EvaluationReport]],
    ):
        grouped: dict[GroupName, list[tuple[TargetMetadata, EvaluationReport]]] = defaultdict(list)
        for metadata, report in reports:
            grouped[metadata.group_name].append((metadata, report))

        for group_name, report_subset in grouped.items():
            scope = AnalysisScope(
                aggregation=AnalysisAggregation.RUN_AND_TARGET,
                group_name=group_name,
            )
            if self.storage.has_analysis_output(scope=scope):
                _logger.info(
                    "Skipping analysis for group %s, already exists",
                    group_name,
                )
                continue

            _logger.info("Running analysis for group comparison")
            run_analysis = self.pipeline.run_for_reports(
                reports=report_subset,
                scope=scope,
            )
            self.storage.save_analysis_output(output=run_analysis)

    def run_for_targets(
        self,
        reports: list[tuple[TargetMetadata, EvaluationReport]],
    ):
        grouped: dict[tuple[GroupName, TargetName], list[tuple[TargetMetadata, EvaluationReport]]] = defaultdict(list)
        for metadata, report in reports:
            grouped[metadata.group_name, metadata.name].append((metadata, report))

        for (group_name, target_name), report_subset in grouped.items():
            scope = AnalysisScope(
                aggregation=AnalysisAggregation.RUN_AND_NONE,
                target_name=target_name,
                group_name=group_name,
            )
            if self.storage.has_analysis_output(scope=scope):
                _logger.info(
                    "Skipping analysis for target %s in group %s, already exists",
                    target_name,
                    group_name,
                )
                continue

            _logger.info("Running analysis for target comparison")
            run_analysis = self.pipeline.run_for_reports(
                reports=report_subset,
                scope=scope,
            )
            self.storage.save_analysis_output(output=run_analysis)


__all__ = ["BenchmarkComparisonPipeline"]
