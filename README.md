# GeospatialAIBot | AI for Safer Roads - ADB Challenge

## Overview

Road traffic crashes remain one of the leading causes of preventable deaths worldwide. While many road authorities establish posted speed limits using engineering guidelines, these limits often become outdated as surrounding environments change.

This project develops an AI-driven framework that evaluates whether posted speed limits align with Safe System principles by combining mobility, geospatial, and contextual datasets.

Rather than detecting speeding behavior, our goal is to determine whether the posted speed limit itself is appropriate for each road segment.

The system identifies locations where current limits may expose vulnerable road users to unnecessary risk and generates actionable recommendations for policymakers.

## Challenge

This project is being developed for the AI for Safer Roads Innovation Challenge, organized by:

- Asian Development Bank (ADB)
- World Bank Development Impact Group
- AI for Good
- International Telecommunication Union (ITU)

### Challenge statement:

How might we use AI and mobility data to determine where speed limits are misaligned with real-world road conditions, supporting evidence-based speed management across Asia and the Pacific?

## Objectives

Our solution aims to:

- Evaluate whether posted speed limits follow Safe System principles
- Detect road segments where limits appear inappropriate
- Identify high-risk areas for vulnerable road users
- Produce an interpretable Speed Safety Score
- Generate geospatial visualizations for policymakers
- Build a scalable workflow applicable across multiple countries

## Features
🚗 Operating speed analysis
🗺️ GIS-based road segmentation
🧠 AI/ML risk prediction
🚸 Vulnerable road user exposure estimation
📍 Interactive risk maps
📊 Speed Safety Score generation
📈 Explainable recommendations for speed limit review

## Data Sources:

The solution is designed to integrate multiple data sources, including:

- GPS probe data
- Posted speed limits
- Operating speed distributions
- Road classifications
- Traffic intensity
- Street-level imagery (Mapillary)
- Land use information
- Population density
- School proximity
- Market proximity
- Additional contextual geographic datasets

(Some datasets are available only through the competition under NDA.)

## Methodology

Raw Mobility Data
        │
        ▼
Feature Engineering
        │
        ▼
Road Context Enrichment
        │
        ▼
Risk Modeling
        │
        ▼
Speed Safety Score
        │
        ▼
Spatial Visualization
        │
        ▼
Policy Recommendations


## Technology Stack
- Python
- Pandas
- GeoPandas
- NumPy
- Scikit-learn
- XGBoost / LightGBM
- PyTorch (optional)
- Shapely
- Rasterio
- Folium
- Plotly
- Matplotlib
- Jupyter Notebook

## Expected Outputs

The project generates:

- Speed Safety Score for each road segment
- Risk classification
- GIS layers
- Interactive maps
- Model explainability reports
- Candidate roads for speed limit review

## Future Enhancements
- Deep learning using street-level imagery
- Graph Neural Networks for road networks
- Real-time mobility data integration
- Automatic policy recommendation engine
- Interactive web dashboard
- Cross-country model transfer learning

## Team
TBD 

## Acknowledgements

Developed as part of the AI for Safer Roads Innovation Challenge organized by the Asian Development Bank in collaboration with the World Bank Development Impact Group, AI for Good, and the International Telecommunication Union (ITU).

## License

This repository is intended for research and competition purposes.
