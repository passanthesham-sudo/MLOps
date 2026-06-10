# Lab 3 — Feature Store Design for Store Sales Forecasting

**Dataset:** Kaggle Store Sales – Time Series Forecasting  
**Goal:** Predict daily unit sales for 33 product families across 54 stores in Ecuador

---

## 1. Dataset Overview

| File | Description | Key Columns |
|---|---|---|
| `train.csv` | Daily store-family sales | date, store_nbr, family, sales, onpromotion |
| `stores.csv` | Store metadata | store_nbr, city, state, type, cluster |
| `oil.csv` | Daily oil price (Ecuador is oil-dependent) | date, dcoilwtico |
| `holidays_events.csv` | National/regional/local holidays | date, type, locale, transferred |
| `transactions.csv` | Daily transaction counts per store | date, store_nbr, transactions |

**Characteristics:**
- Multi-entity time series: 54 stores × 33 product families × ~1700 days
- Hierarchical geography: store → city → state
- Strong external signals: oil price shocks, holidays, promotions
- Heavy data leakage risk — features must respect time boundaries

---

## 2. Data Pipeline Methodology

```
Raw Sources
    │
    ▼
[Ingestion]   Pull CSVs from Kaggle / object storage
    │
    ▼
[Validation]  Schema checks, null audits, range checks on sales ≥ 0,
              oil price continuity, holiday date uniqueness
    │
    ▼
[Joining]     Merge stores → sales (on store_nbr)
              Merge oil → sales (on date, forward-fill missing prices)
              Merge holidays → sales (on date + locale match)
              Merge transactions → sales (on date + store_nbr)
    │
    ▼
[Feature      See Section 3
 Engineering]
    │
    ▼
[Feature      Write to Feature Store
 Store Write] (offline: Parquet/Delta; online: Redis/DynamoDB)
    │
    ▼
[Training     Point-in-time join: retrieve features ≤ label date
 Dataset]     to prevent future leakage
```

**Point-in-Time Correctness** is critical here. When constructing the training set for date `t`, all lag features and rolling statistics must be computed from data available strictly before `t`. Failure to enforce this is the most common source of inflated offline metrics that don't translate to production.

---

## 3. Feature Engineering

### 3.1 Temporal Features (entity: date)
| Feature | Description |
|---|---|
| `day_of_week`, `month`, `year` | Calendar decomposition |
| `is_weekend` | Saturday/Sunday flag |
| `day_of_year`, `week_of_year` | Seasonality encoding |
| `days_to_next_holiday`, `days_since_last_holiday` | Holiday proximity |
| `is_holiday`, `holiday_type` | Local/national/regional flag |
| `is_transferred_holiday` | Transferred holiday flag |

### 3.2 Sales Lag & Rolling Features (entity: store × family × date)
| Feature | Description |
|---|---|
| `sales_lag_7`, `_14`, `_28` | Past sales at 1/2/4-week offsets |
| `sales_roll_mean_7`, `_28` | Rolling mean sales |
| `sales_roll_std_7` | Rolling volatility |
| `promo_lag_7`, `promo_roll_14` | Past promotion counts |

### 3.3 Store Features (entity: store)
| Feature | Description |
|---|---|
| `store_type` | A/B/C/D/E store classification |
| `store_cluster` | Cluster ID (1–17) |
| `city`, `state` | Geographic encoding |
| `avg_transactions_30d` | Rolling avg daily transactions |

### 3.4 External Features (entity: date)
| Feature | Description |
|---|---|
| `oil_price`, `oil_price_lag_1` | Current and yesterday's price |
| `oil_price_pct_change_7` | 7-day price momentum |
| `oil_price_roll_mean_30` | 30-day smoothed price |

---

## 4. Feature Store Design

### 4.1 Architecture

```
                 ┌─────────────────────────────────┐
                 │         FEATURE STORE           │
                 │                                 │
  Batch ETL ───▶ │  Offline Store (Parquet/Delta)  │◀─── Historical
  (daily/weekly) │  • All feature groups           │     training jobs
                 │  • Point-in-time correct reads  │
                 │                                 │
  Streaming ───▶ │  Online Store (Redis)           │◀─── Real-time
  (oil feed)     │  • Latest store features        │     inference
                 │  • Latest oil price             │
                 └─────────────────────────────────┘
```

### 4.2 Feature Groups

| Group | Update Frequency | Storage | Entities |
|---|---|---|---|
| `store_static_fg` | Monthly | Offline only | store_nbr |
| `time_calendar_fg` | Pre-computed | Offline only | date |
| `holiday_fg` | Yearly + ad-hoc | Offline only | date |
| `sales_lag_fg` | Daily (T+1) | Offline + Online | store_nbr, family, date |
| `oil_price_fg` | Daily | Offline + Online | date |
| `transaction_fg` | Daily | Offline + Online | store_nbr, date |

### 4.3 Key Design Decisions

**Offline-first, selective online.** Static features (store type, cluster) and pre-computed calendars never need online serving. Only the features needed for real-time predictions (latest lag features, current oil price) are mirrored to the online store, reducing cost.

**Lag window selection.** Weekly (7-day) and monthly (28-day) lags capture both short-term demand shocks and seasonal baselines without introducing leakage.

**Promotion handling.** `onpromotion` is a known future value (stores know their promotion schedule). It can be included as a future-known feature without leakage — this is an important distinction from sales lags.

**Holiday locale matching.** Ecuador holidays are national, regional, or local. A feature join must match `locale_name` to `city`/`state`/`Ecuador` to avoid incorrect holiday flags for stores in non-affected regions.

---

## 5. Summary

The Store Sales dataset requires a multi-entity, time-aware Feature Store. The most important design principle is **point-in-time correctness** — every feature must be computed from data available at or before the prediction date. The recommended architecture splits features into a daily-updated offline store (for training) and a selectively mirrored online store (for inference), with feature groups organized by update frequency and entity type.

---

*References: Feast documentation, Hopsworks Feature Store paper, Kaggle Store Sales EDA notebooks.*
