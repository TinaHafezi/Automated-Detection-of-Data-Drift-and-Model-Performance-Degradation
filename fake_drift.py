import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, path: str = "config.yaml"):
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys, default=None):
        """Get nested config value with optional default."""
        value = self.config
        for k in keys:
            if k not in value:
                if default is not None:
                    return default
                raise KeyError(f"Missing key in config: {'/'.join(keys)}")
            value = value[k]
        return value


def inject_telco_drift(input_path: Path, config: Config, output_path: Optional[Path] = None,
                       seed: int = 42) -> pd.DataFrame:
    """Inject synthetic drift into Telco customer churn data using config parameters."""
    np.random.seed(seed)

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded Telco data from {input_path} with shape {df.shape}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise

    drift_cfg = config.get("drift", "telco")

    # 1. Distribution Drift
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] *= drift_cfg.get("monthly_charges_multiplier", 1.0)
    if "tenure" in df.columns:
        df["tenure"] *= drift_cfg.get("tenure_multiplier", 1.0)

    # 2. Category Drift
    if "Contract_One year" in df.columns and "Contract_Two year" in df.columns:
        df["Contract_One year"] = drift_cfg.get("contract_one_year_value", 1)
        df["Contract_Two year"] = drift_cfg.get("contract_two_year_value", 0)

    # 3. Data Quality Drift
    if "TotalCharges" in df.columns:
        frac = drift_cfg.get("total_charges_nan_fraction", 0.0)
        if frac > 0:
            nan_indices = df.sample(frac=frac).index
            df.loc[nan_indices, "TotalCharges"] = np.nan

    # 4. Label Noise
    if "Churn" in df.columns:
        frac = drift_cfg.get("churn_noise_fraction", 0.0)
        if frac > 0:
            noise_indices = df.sample(frac=frac).index
            df.loc[noise_indices, "Churn"] = 1

    output_path = output_path or input_path.with_name("current_drifted.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Telco drift injected! Saved to {output_path}")

    return df


def inject_ethereum_drift(input_path: Path, config: Config, output_path: Optional[Path] = None,
                          seed: int = 42) -> pd.DataFrame:
    """Inject synthetic drift into Ethereum price data using config parameters."""
    np.random.seed(seed)

    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded Ethereum data from {input_path} with shape {df.shape}")
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_path}")
        raise

    drift_cfg = config.get("drift", "ethereum")

    # 1. Price Shock
    if "close" in df.columns:
        df["close"] *= drift_cfg.get("close_multiplier", 1.0)
    if "volume" in df.columns:
        df["volume"] *= drift_cfg.get("volume_multiplier", 1.0)

    # 2. Volatility Spike
    if all(col in df.columns for col in ["high", "low"]):
        df["high"] *= drift_cfg.get("high_multiplier", 1.0)
        df["low"] *= drift_cfg.get("low_multiplier", 1.0)

    # 3. Data Quality Issues
    if "volume" in df.columns:
        frac = drift_cfg.get("volume_zero_fraction", 0.0)
        if frac > 0:
            zero_indices = df.sample(frac=frac).index
            df.loc[zero_indices, "volume"] = 0
    if "close" in df.columns:
        frac = drift_cfg.get("close_nan_fraction", 0.0)
        if frac > 0:
            nan_indices = df.sample(frac=frac).index
            df.loc[nan_indices, "close"] = np.nan

    # 4. Target Corruption
    if "target" in df.columns:
        df["target"] = df["target"] * np.random.uniform(
            drift_cfg.get("target_min_multiplier", 1.0),
            drift_cfg.get("target_max_multiplier", 1.0),
            size=len(df)
        )

    output_path = output_path or input_path
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Ethereum drift injected! Saved to {output_path}")

    return df


def main():
    SEED = 42
    np.random.seed(SEED)

    config = Config("config.yaml")
    dataset = config.get("app", "dataset_name")

    if dataset == "telco":
        input_path = Path("Train model/current.csv")
        inject_telco_drift(input_path, config, seed=SEED)
    elif dataset == "ethereum":
        input_path = Path("Eth/current.csv")
        inject_ethereum_drift(input_path, config, seed=SEED)
    else:
        logger.error(f"Unknown dataset name: {dataset}")
        return

    logger.info("Drift injection completed successfully")


if __name__ == "__main__":
    main()
