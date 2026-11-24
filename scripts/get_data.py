
#!/usr/bin/env python3

import logging
import subprocess
from pathlib import Path


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


S3_BASE = "s3://open-targets-public-data-releases/platform/26.09/output"


def download_s3_directory(
    s3_path: str,
    destination: Path,
):
    """Download a public S3 directory recursively."""

    destination.mkdir(parents=True, exist_ok=True)

    command = [
        "aws",
        "s3",
        "cp",
        "--recursive",
        "--no-sign-request",
        s3_path,
        str(destination),
    ]

    logger.info("Downloading from S3: %s", s3_path)
    logger.info("Destination: %s", destination)

    try:
        subprocess.run(command, check=True)

        logger.info("Download completed successfully.")

    except FileNotFoundError:
        logger.exception(
            "AWS CLI not found. Install AWS CLI and ensure it is in PATH."
        )
        raise

    except subprocess.CalledProcessError:
        logger.exception("S3 download failed.")
        raise


def main():

    # Create repository directories
    directories = [
        Path("data"),
        Path("output")
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directory ready: %s", directory)

    # Download disease dataset
    disease_dir = Path("data/disease")

    if not disease_dir.exists():
        download_s3_directory(
            f"{S3_BASE}/disease/",
            disease_dir,
        )
    else:
        logger.info("Disease dataset already exists. Skipping.")

    # Example: download association_overall_direct


    

if __name__ == "__main__":
    main()
