# BookWise Exploratory Data Analysis
- Source dimensions: 21 rows x 12 columns
- Cleaned rows: 21 (removed 0 records)
- Exact duplicate source rows: 0
- Duplicate ISBN-13 values: 0
- Duplicate ISBN-10 values: 0
- Duplicate normalized titles: 0

## Missing values in source data
- subtitle: 17 (81.0%)

## Numeric distributions
- published_year: valid=21, min=1813, median=1988.0, max=2021
- average_rating: valid=21, min=3.9, median=4.5, max=4.9
- num_pages: valid=21, min=180, median=336.0, max=864
- ratings_count: valid=21, min=980000, median=4200000.0, max=52000000

## Text distributions
- Distinct non-empty categories: 18
- Distinct non-empty authors: 19

## Cleaning decisions
- Read ISBN-10 and ISBN-13 as strings, preserving leading zeroes.
- Normalized whitespace and converted empty textual fields to missing values.
- Kept records with missing optional text fields; invalid numeric ranges become missing, not fabricated values.
- Removed exact duplicates and then duplicate ISBN-13/title keys deterministically, retaining the last sorted record.