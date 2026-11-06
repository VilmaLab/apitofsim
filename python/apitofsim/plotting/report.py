class UnknownReportTypeError(ValueError):
    pass


def get_report(db, report_type):
    from typing import List

    import pandas

    if not db.is_experiment_db() and report_type in {
        "event-report",
        "experiment-pathway-report",
        "experiment-cluster-report",
        "experiment-summary",
        "spectrogram",
    }:
        raise UnknownReportTypeError(
            f"Report type {report_type} is only available for experiment databases"
        )

    if report_type == "spectrogram":
        from apitofsim.plotting.spectrogram import get_intensities

        dataframes: List[pandas.DataFrame] = []
        for row in db.report_df("experiment_summary").itertuples():
            dataframes.append(
                get_intensities(
                    db,
                    experiment_id=row.experiment_run_id,
                    is_single_pathway=row.is_single_pathway,
                )
            )
        df = pandas.concat(dataframes)
        return df
    report = db.db.table(report_type.replace("-", "_")).df()
    if report_type == "event-report" and report.empty:
        raise UnknownReportTypeError("No recorded events in the database")
    return report
