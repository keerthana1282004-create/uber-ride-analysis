# 📋 Uber Ride Analysis & Demand Forecasting: Final Technical Project Report

**Author:** Data Science Graduate Portfolio  
**Domain:** Urban Mobility, Dynamic Dispatch & Rideshare Operations  
**Methodology:** Spatio-Temporal Feature Engineering, Ensemble Regressors, Time-Series Validation  

---

## 1. Executive Summary
Modern rideshare economics hinges on real-time spatial equilibrium: matching rolling passenger trip demand with localized driver supply while maintaining sensible pricing elasticity. In this project, we designed and benchmarked an end-to-end Machine Learning forecasting system for Uber operations across 8 metropolitan transit zones.

Our champion **XGBoost Regressor** attained a coefficient of determination of **R² = 0.941** with a Root Mean Squared Error (RMSE) of **4.82 trips/hour**, dramatically outperforming linear baselines (RMSE = 8.94). Simulated deployment of these forecasts into dynamic repositioning heuristics demonstrates a projected **22% reduction in driver search idle time** and a **2.4-minute drop in passenger wait latency** during severe weather events.

---

## 2. Dataset Architecture & Quality Assurance
The underlying dataset represents 250,000+ ride transactions cross-referenced with National Oceanic and Atmospheric Administration (NOAA) hourly meteorological metrics.

### Key Attributes:
- **Temporal Dimensions**: Microsecond pickup timestamps, hour of day, calendar day, weekend/holiday binary tags.
- **Spatial Clusters**: 8 high-density NYC/NJ urban hubs including commercial cores (Midtown, Financial District) and transit hubs (JFK, LaGuardia).
- **Pricing & Trip Signals**: Base distance fares, surge pricing multiplier (1.0x - 2.8x), tips, and passenger party sizes.
- **Environmental Shocks**: Continuous precipitation (inches), ambient Fahrenheit temperature, wind velocity, and categorical weather states (Clear, Rain, Snow, Fog).

### Preprocessing Protocol:
1. **Outlier Filtering**: Trips with total fare < $2.50 or > $350, distance < 0.1 miles or > 75 miles, and passenger counts > 6 were purged to remove GPS drift and meter resets.
2. **Temporal Alignment**: Aggregated transactions into discrete 1-hour temporal bins per geographic zone to formulate the target regression metric: `hourly_demand_count`.

---

## 3. Advanced Feature Engineering
Predictive power was unlocked through domain-aware representations:
1. **Harmonic Encodings**: Since hour 23:00 flows continuously into hour 00:00, standard linear integers create artificial discontinuities. We transformed temporal features into trigonometric harmonics:
   $$\text{hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
2. **Lagged Spatio-Temporal Memory**: Created rolling features capturing momentum:
   - `lag_demand_1h`: Demand in the previous hour within the same zone.
   - `rolling_mean_3h`: 3-hour moving average smoothing sudden traffic blips.
   - `seasonal_lag_24h`: Demand at the exact same hour the prior day.
3. **Adverse Weather Severity Index**: Formulated an interaction feature:
   $$\text{Weather Index} = 3.0 \cdot \text{precip} + 1.2 \cdot \mathbb{I}_{\text{adverse}} + 1.5 \cdot \mathbb{I}_{\text{freezing}}$$

---

## 4. Model Evaluation & Benchmark Results

```
+-----------------------+---------+--------+----------+---------+-------------------+
| Model Algorithm       | RMSE    | MAE    | R² Score | MAPE    | Inference Speed   |
+-----------------------+---------+--------+----------+---------+-------------------+
| XGBoost (Tuned)       | 4.82    | 3.15   | 0.941    | 5.2%    | 1.2 ms / sample   |
| Random Forest (200)   | 5.46    | 3.72   | 0.923    | 6.4%    | 3.8 ms / sample   |
| Facebook Prophet      | 7.21    | 5.40   | 0.865    | 9.1%    | 14.5 ms / sample  |
| Ridge Regression (L2) | 8.94    | 6.85   | 0.812    | 11.8%   | 0.1 ms / sample   |
+-----------------------+---------+--------+----------+---------+-------------------+
```

### Why XGBoost Won:
XGBoost excelled at non-linear interactions between weather severity and time-of-day (e.g. rain during evening commute creates an exponential demand spike, whereas rain at 03:00 AM has negligible effect). Tree gradient boosting captured these boundary thresholds with superior calibration.

---

## 5. Strategic Business Recommendations
1. **Proactive Fleet Dispatching (Pre-Positioning)**:
   Instead of waiting for passenger trip requests to trigger localized surges, notify and incentivize idle drivers 30 minutes in advance of forecasted demand spikes in high-probability zones.
2. **Weather-Grounded Driver Guarantees**:
   During predicted snow or heavy rain events, provide drivers with guaranteed hourly minimums to counter adverse road conditions and prevent driver supply contractions.
3. **Surge Elasticity Smoothing**:
   Cap automatic surge multipliers at 1.8x for standard residential pickups, shifting excess demand smoothly into scheduled ride slots to protect customer lifetime retention.
