# Cross-Domain Aspect-Based Sentiment Analysis

A research-oriented project exploring Aspect-Based Sentiment Analysis (ABSA), with a focus on cross-domain adaptation under limited labeled data.

## Overview

Aspect-Based Sentiment Analysis aims to identify sentiment toward specific aspects within a sentence.

For example:

> The battery life is great but the screen is expensive.

The sentiment toward each aspect is different:

- **battery** → Positive
- **screen** → Negative

This project starts with a simple rule-based baseline and will gradually extend to machine learning and Transformer-based approaches for cross-domain ABSA.

## Current Implementation

The current version includes:

- A rule-based ABSA baseline
- Aspect-specific local context extraction
- Positive, negative, and neutral sentiment classification
- CSV-based evaluation
- A small sample dataset for testing

## Project Structure

```text
cross-domain-absa/
├── data/
│   └── sample_absa.csv
├── notebooks/
├── results/
├── src/
│   └── baseline.py
├── .gitignore
├── LICENSE
└── README.md