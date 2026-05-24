import csv
import random
import sys
from pathlib import Path

"""Synthetic data generator for stunting_wasting_dataset-like CSV.

This script creates a CSV with the same columns as data3/stunting_wasting_dataset.csv:
Jenis Kelamin,Umur (bulan),Tinggi Badan (cm),Berat Badan (kg),Stunting,Wasting

Usage:
    python generate_synthetic_data.py  # creates data3/stunting_wasting_dataset_100k.csv
"""

GENDERS = ["Laki-laki", "Perempuan"]
STUNTING_STATUSES = [
    "Severely Stunted",
    "Stunted",
    "Normal",
    "Tall",
]
WASTING_STATUSES = [
    "Severely Underweight",
    "Underweight",
    "Normal weight",
    "Risk of Overweight",
]


def sample_child() -> dict:
    gender = random.choice(GENDERS)
    age_months = random.randint(0, 60)

    # Rough height and weight patterns by age
    # Base medians (approximate, loosely based on WHO) + noise
    if age_months <= 3:
        base_height = 55 + age_months * 2.0
        base_weight = 4.0 + age_months * 0.6
    elif age_months <= 12:
        base_height = 60 + age_months * 1.5
        base_weight = 6.0 + age_months * 0.4
    elif age_months <= 24:
        base_height = 75 + (age_months - 12) * 0.8
        base_weight = 9.0 + (age_months - 12) * 0.25
    else:
        base_height = 87 + (age_months - 24) * 0.7
        base_weight = 11.0 + (age_months - 24) * 0.2

    height_cm = round(random.normalvariate(base_height, 3.0), 1)
    weight_kg = round(random.normalvariate(base_weight, 1.2), 1)

    # Assign stunting / wasting qualitatively based on rough z-like deviation
    height_dev = (height_cm - base_height) / 3.0
    weight_dev = (weight_kg - base_weight) / 1.2

    # Stunting status
    if height_dev < -1.8:
        stunting = "Severely Stunted"
    elif height_dev < -0.9:
        stunting = "Stunted"
    elif height_dev > 1.2:
        stunting = "Tall"
    else:
        stunting = "Normal"

    # Wasting status (weight-for-height proxy)
    if weight_dev < -1.8:
        wasting = "Severely Underweight"
    elif weight_dev < -0.9:
        wasting = "Underweight"
    elif weight_dev > 1.2:
        wasting = "Risk of Overweight"
    else:
        wasting = "Normal weight"

    return {
        "Jenis Kelamin": gender,
        "Umur (bulan)": age_months,
        "Tinggi Badan (cm)": height_cm,
        "Berat Badan (kg)": weight_kg,
        "Stunting": stunting,
        "Wasting": wasting,
    }


def main(n_rows: int = 100_000) -> None:
    root = Path(__file__).parent
    data_dir = root / "data3"
    data_dir.mkdir(exist_ok=True)

    # Allow overriding n_rows from command line: python add_data.py 800000
    if len(sys.argv) > 1:
        try:
            n_rows = int(sys.argv[1])
        except ValueError:
            print(f"Invalid row count '{sys.argv[1]}', using default {n_rows}")

    out_path = data_dir / f"stunting_wasting_dataset_{n_rows}.csv"

    fieldnames = [
        "Jenis Kelamin",
        "Umur (bulan)",
        "Tinggi Badan (cm)",
        "Berat Badan (kg)",
        "Stunting",
        "Wasting",
    ]

    random.seed(42)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(n_rows):
            writer.writerow(sample_child())

    print(f"Generated synthetic dataset with {n_rows} rows at: {out_path}")


if __name__ == "__main__":
    main()
