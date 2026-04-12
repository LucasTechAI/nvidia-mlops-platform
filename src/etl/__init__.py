"""ETL (Extract, Transform, Load) modules for stock data."""

from .extractor_nvidia import extract_nvidia_data as extract_nvidia_data


def refresh_stock_data() -> bool:
    """Download the latest NVDA data from Yahoo Finance and reload into SQLite.

    This is a convenience wrapper used by the training and
    Champion-Challenger pipelines so the model always trains on the
    most recent market data.

    Returns:
        True if the refresh succeeded, False otherwise.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        from src.etl.extractor_nvidia import extract_nvidia_data, save_data
        from src.etl.load_sqlite_nvidia import load_csv_to_sqlite

        logger.info("=" * 60)
        logger.info("ETL: Refreshing NVDA stock database …")
        logger.info("=" * 60)

        df = extract_nvidia_data(period="max", interval="1d")
        if df is None or df.empty:
            logger.warning("ETL: No data returned from Yahoo Finance — skipping refresh")
            return False

        save_data(df, "nvidia_stock.csv")
        load_csv_to_sqlite()

        last_date = df["Date"].max()
        logger.info("ETL: Database refreshed — %d records, last date: %s", len(df), last_date)
        return True

    except Exception as exc:
        logger.error("ETL: Failed to refresh stock data — %s", exc)
        return False
