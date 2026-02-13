# Predicting Fair Rental Price & Investment Return (Malaysia)

A decision-support system for Malaysia’s rental market that predicts **fair monthly rent**, estimates **property investment returns**, and provides **geo-search**  for tenants and investors.  
It also improves market understanding through **model explainability through SHAP feature impact analysis**.

## Key Features
- Rental price prediction (ML regression)
- Investment metrics analysis dashboard (ROI-related metrics)
- Investment worthiness explanation report (clear rationale & insights)
- Investment suggestions using fuzzy logic
- Map-based nearby public transport search (MRT/LRT/bus + distance)

## Dataset
Combined dataset (~20k records) from public sources + scraped listings (Kaggle + mudah.com / murah.com listings). 

## Flow
1. Define scope & target outputs (rent prediction + ROI + geo-search)
2. Scrape rental listings & collect external data (transport/location)
3. Store and organize data (database / structured files)
4. Clean data & engineer features (encoding, location features, outliers)
5. Train & evaluate rent prediction models (select best)
6. Explain model with SHAP (key factors + insights)
7. Build investment module (metrics + worthiness report)
8. Implement fuzzy logic for investment suggestions
9. Develop geo-search (nearby MRT/LRT/bus + distance)
10. Integrate into UI and test end-to-end

## Tech Stack
- Python (Pandas, Scikit-learn, XGBoost/LightGBM, SHAP)
- Node-RED + Selenium (data scraping)
- PostgreSQL (data storage)
- MATLAB App Designer (UI / dashboard / deployment)
  
## Author
LEE SHIN YEE 
