# Data sources, bias and join strategy

## Sources
| # | Dataset | URL | Records used | Mode this run |
|---|---------|-----|-------------:|---------------|
| 1 | Global Air Pollution Dataset | https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset | 23,463 | REAL Kaggle file (global air pollution dataset.csv) |
| 2 | Air Quality Data in India | https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india | 29,531 | REAL Kaggle file (city_day.csv) |
| 3 | Asthma Disease Dataset | https://www.kaggle.com/datasets/rabieelkharoua/asthma-disease-dataset | 10,000 | REAL Kaggle file (asthma_disease_data.csv), 2,392 rows |
| 4 | Open-Meteo Air Quality + Weather API | https://air-quality-api.open-meteo.com/v1/air-quality , https://api.open-meteo.com/v1/forecast | 11,120 | live Open-Meteo API |

## Known biases
- **Urban sensor skew:** monitoring stations sit mostly in large cities; rural exposure is under-represented.
- **Winter smog seasonality:** PM2.5 peaks October-January in the Indo-Gangetic plain; a three-month window over-weights one season.
- **Self-reported clinical bias:** symptoms, exposure and allergy fields are self-reported and can be under- or over-stated.
- **Cross-country transfer:** the Indian and global datasets are used for seasonal context, not as a direct measurement of Pakistani exposure.
- **Proxy label:** the risk tier is rule-based plus noise, not a recorded clinical outcome.

## Join strategy
Spatial key (city) + temporal key (hour) link a patient-day to the Open-Meteo timeline; Datasets 1-2 provide AQI reference and seasonal baselines;
the clinical risk tier combines patient profile and ambient exposure. See the notebook markdown above for the full description.
