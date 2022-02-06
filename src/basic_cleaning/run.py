#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download the artifact and read it into a pandas dataframe
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Perform some basic cleaning
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info(
        f"Basic cleaning: dropping outlier price values outside of the range [{args.min_price}, {args.max_price}]")
    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info(
        "Basic cleaning: converting the column last_review to the datetime format")

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Write the data to disk
    df.to_csv("clean_sample.csv", index=False)

    # Create a new artifact and load it to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The name of the input artifact containing the raw data",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name of the output artifact that will contain the clean data",
        required=True)

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price to consider",
        required=True
    )

    args = parser.parse_args()

    go(args)
