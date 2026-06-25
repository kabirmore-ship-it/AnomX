# AnomX — Anomaly Detection in Financial Trading Events

A project to detect suspicious and fraudulent behaviour in a synthetic financial trading platform by generating realistic event data, engineering meaningful features, and building the foundation for a machine learning anomaly detection pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Python, Git & Foundations](#python-git--foundations)
- [Event Generation](#event-generation)
- [Feature Engineering & EDA](#feature-engineering--eda)
- [Repository Structure](#repository-structure)
---

## Project Overview

AnomX is an anomaly detection system designed for a simulated forex and crypto trading platform. The platform generates thousands of user events — logins, trades, deposits, withdrawals, session activity, and KYC updates some of which are intentionally injected with anomalous patterns that mirror real-world financial fraud.

The core objectives of this project are:

- Generate realistic synthetic event data with labelled anomalies.
- Engineer features from raw event logs that make anomalies detectable by machine learning models.
- Build a foundation for training and evaluating anomaly detection classifiers.

This project mirrors challenges faced by compliance and fraud teams at real financial institutions, where suspicious behaviour must be identified quickly and accurately across large volumes of transactional data.

---

### Why I chose this project

I wanted to work on a project that involved more than just training a machine learning model. In real-world fraud detection, the hardest part is often generating meaningful signals from noisy event data.

This project helped me practice:

- Designing realistic datasets
- Creating behavioural features from user activity
- Working with time-series event logs
- Thinking about fraud detection from a business perspective


## Python, Git & Foundations
### Development Foundations

The first stage of the project focused on refreshing Python, Pandas, NumPy, and Git workflows that would later be used throughout the pipeline.

Key areas covered:

- Python scripting and data processing
- Pandas and NumPy operations
- Git and GitHub workflows
- Repository organisation and documentation

These skills were then applied to the event generation and feature engineering components of the project.

## Event Generation

### How Synthetic Event Data Was Generated

Synthetic data was generated using `src/data_gen/generate_events.py`. The pipeline works in three stages:

**Stage 1 — Build User Profiles**

A pool of `N_USERS` users is created, each with a persistent profile containing:

- A home country and corresponding timezone offset
- A preferred device and home IP address
- Typical trade volume and deposit amount (drawn from random distributions)
- A flag marking whether they are anomalous, and if so, which anomaly type they exhibit

A configurable fraction of users (`anomaly_fraction` in `config.yaml`) are designated as anomalous and assigned one of ten fraud patterns.

**Stage 2 — Inject Anomaly Patterns**

For every anomalous user, a dedicated generator function produces a burst of labelled events that encode their specific fraud behaviour. These events are injected first to guarantee representation.

**Stage 3 — Fill Remaining Events**

Normal events are generated at random to reach the target `N_EVENTS` count. Even anomalous users generate some normal-looking events — reflecting how real fraudsters blend into background activity.

The final dataset is sorted by timestamp and saved as `data/raw/events.csv`.

### Types of Events

Six event types are generated, weighted to reflect realistic platform activity:

| Event Type   | Weight | Description |
|--------------|--------|-------------|
| `trade`      | 35%    | Buy/sell orders with instrument, volume, PnL, and duration |
| `login`      | 22%    | Authentication events with IP, country, device, and success flag |
| `deposit`    | 15%    | Funds added to account via card, bank, or crypto |
| `session`    | 12%    | Platform browsing sessions with duration and click counts |
| `withdrawal` | 10%    | Funds removed from account |
| `kyc_change` | 6%     | Identity document or contact detail updates |

### Ten Anomaly Patterns

Each pattern is designed to encode a distinct fraud signal in the data:

| Anomaly Type | Behaviour | Key Signal |
|---|---|---|
| `ip_hopper` | Logs in from multiple countries within minutes | High `timezone_gap_hours` between consecutive logins |
| `wash_trader` | Trades same instrument at 10–20x normal volume, always profits | Extreme `trade_volume_vs_baseline`; consistently positive `pnl` |
| `deposit_withdrawal_cycler` | Deposits then immediately withdraws nearly the same amount | `is_immediate_withdrawal = 1`; crypto method |
| `bot_trader` | Sessions at 2–4 AM with hundreds of clicks per minute | `click_rate_per_min` in the range of 200–3000 |
| `structurer` | Makes many deposits just under 1,000 to avoid reporting thresholds | `amount` clustered between 490 and 999 |
| `brute_forcer` | Escalating failed login attempts followed by sudden success | Rising `failed_attempts` then `login_success = 1` |
| `dormant_withdrawer` | Account inactive for months, then large crypto withdrawal | High `account_age_days`; very large `amount` |
| `consistent_winner` | Sub-10-second trades with 100% positive PnL | `trade_duration_seconds < 10`; always positive `pnl` |
| `device_switcher` | Different device fingerprint on every login | High `unique_devices_last_10_logins` |
| `kyc_manipulator` | Updates identity details then withdraws large sums within the hour | KYC event immediately followed by high-value `withdrawal` |

### Dataset Structure and Key Insights

The generated dataset contains 35 columns covering raw event attributes plus several pre-computed signals:

- **Temporal**: `timestamp`, `hour_of_day`, `day_of_week`, `is_weekend`
- **Login**: `ip_address`, `country`, `device`, `login_success`, `failed_attempts`, `timezone_gap_hours`
- **Trade**: `instrument`, `lot_size`, `trade_volume`, `pnl`, `margin_used`, `trade_duration_seconds`, `trade_volume_vs_baseline`, `is_night_trade`
- **Financial**: `amount`, `method`, `is_immediate_withdrawal`
- **Session**: `session_duration_mins`, `page_clicks`, `click_rate_per_min`
- **KYC**: `kyc_change_type`
- **Labels**: `is_anomalous`, `anomaly_type`

A notable design choice is that several signals are pre-computed at generation time (e.g. `timezone_gap_hours`, `trade_volume_vs_baseline`, `click_rate_per_min`) to serve as ground-truth anchors during feature validation.

---

## Feature Engineering & EDA

### Features Created from Raw Event Logs

Feature engineering is handled in `src/features/feature_engineering.py`. All features are derived from the raw event log without using the `is_anomalous` label — simulating a real unsupervised or semi-supervised detection scenario.

**Time-Delta Features**

| Feature | Description |
|---|---|
| `time_since_last_event_sec` | Seconds elapsed since the user's previous event of any type |
| `time_since_last_login_sec` | Seconds since the user's most recent login |
| `time_since_last_deposit_sec` | Seconds since the user's most recent deposit |

Short gaps are a strong signal for bots and brute-forcers. Dormant withdrawers will show extremely large `time_since_last_login_sec` values.

**Rolling Window Features** (computed over windows of size 3 and 10)

| Feature | Description |
|---|---|
| `roll_N_trade_vol_mean` | Average trade volume over last N trades |
| `roll_N_trade_vol_std` | Volatility of trade volume — low std with high mean signals wash trading |
| `roll_N_pnl_mean` | Average PnL over last N trades — consistently positive is suspicious |
| `roll_N_click_rate_mean` | Average clicks-per-minute over last N sessions |

**Burst Count Features**

| Feature | Description |
|---|---|
| `burst_count_5min` | Number of events this user generated in the last 5 minutes |
| `burst_count_30min` | Number of events in the last 30 minutes |

High burst counts flag brute-forcers (many logins in seconds), bot traders (many sessions in minutes), and deposit-withdrawal cyclers.

**Login Anomaly Features**

| Feature | Description |
|---|---|
| `unique_ips_last_10_logins` | Count of distinct IPs across last 10 logins |
| `unique_countries_last_10_logins` | Count of distinct countries across last 10 logins |
| `unique_devices_last_10_logins` | Count of distinct devices across last 10 logins |
| `rolling_failed_attempts_5` | Total failed login attempts across last 5 login events |

These features directly target `ip_hopper`, `device_switcher`, and `brute_forcer` patterns.

**Deposit/Withdrawal Ratio Features**

| Feature | Description |
|---|---|
| `roll_5_deposit_sum` | Total deposits over last 5 financial events |
| `withdrawal_to_deposit_ratio` | Withdrawal amount relative to recent deposits — high values signal cycling |

**Z-Score Features**

| Feature | Description |
|---|---|
| `trade_vol_zscore` | How many standard deviations this trade's volume is from the user's mean |
| `pnl_zscore` | Z-score of PnL relative to user's own history |
| `amount_zscore` | Z-score of deposit/withdrawal amount relative to user baseline |
| `session_duration_zscore` | Z-score of session length — very short sessions flag bots |

Z-scores are computed per-user, meaning the model learns what is normal *for that individual*, not for the population. This is essential for catching anomalies that would look normal at a platform level.

### Feature Engineering Approach

The objective of feature engineering was to convert raw event logs into behavioural signals usable by anomaly detection models.

Rather than relying on individual events, many features were designed to capture patterns across time. Examples include rolling statistics, event burst counts, login behaviour trends, and transaction ratios.

Several features are calculated based on each user's historical behaviour. This helps identify unusual activity for a specific user rather than comparing everyone against a single global baseline.

The resulting feature set is designed to make fraud patterns more distinguishable while remaining realistic enough to reflect how monitoring systems operate in production environments.

### Exploratory Data Analysis (EDA)

Example Generated Events

The generated dataset contains a mixture of login, trading, session, deposit, withdrawal, and KYC-related events. Each event type populates a different subset of fields depending on the activity being recorded.

Example records from the raw dataset:

EVT_183607,USER_0491,login,2024-03-29 20:36:37,20,4,0,0,none,10.118.233.1,AE,android_app,1.0,0.0,0.0,,,,,,,,,,,,,,,,138
EVT_603409,USER_0313,trade,2024-03-29 20:42:12,20,4,0,0,none,,,,,,,GBPUSD,1.78,15301.37,5.75,3422.56,986.0,1.062,1.0,,,,,,,,189
EVT_346891,USER_0472,session,2024-03-29 20:44:04,20,4,0,0,none,,,firefox_linux,,,,,,,,,,,,,,,19.4,68.0,3.51,,189
EVT_313795,USER_0038,deposit,2024-03-29 20:49:10,20,4,0,0,none,,,,,,,,,,,,,,,208.62,bank,0.0,,,,,239
EVT_940912,USER_0447,login,2024-03-29 20:49:58,20,4,0,0,none,10.75.51.1,US,firefox_l

Key observations from the generated and processed dataset:

- **Class imbalance is significant**: anomalous events make up a minority of the dataset, consistent with real fraud rates. Any model trained on this data must account for this by using class weighting, oversampling (e.g., SMOTE), or threshold tuning.

- **Anomaly patterns cluster in feature space**: `bot_trader` events show `click_rate_per_min` values two orders of magnitude above normal users. `wash_trader` events have `trade_volume_vs_baseline` of 10–20x. These separations confirm the features capture the intended signals.

- **Timezone gaps are a strong login signal**: normal users have a `timezone_gap_hours` of 0 (they always log in from their home country). `ip_hopper` events frequently show gaps of 8–13 hours — reflecting impossible travel between geographically distant countries within minutes.

- **Time-since-last-event reveals bot patterns**: `bot_trader` sessions arrive every 3 minutes like clockwork. Normal user sessions are irregular and spaced hours or days apart. The `burst_count_5min` feature is particularly discriminative here.

- **Structurers are subtle**: amounts between 490 and 999 overlap with normal small deposits. This pattern is only detectable in aggregate — looking at the distribution of a single user's deposit amounts over time, not at individual events.

### Challenges and Lessons Learned

A major challenge was designing realistic normal behaviour before introducing anomalies. If fraudulent behaviour is too obvious, the dataset becomes less useful for evaluating detection methods.

Some key lessons from the project include:

- Behavioural patterns are often more informative than individual events.
- Feature engineering can have a larger impact on detection quality than model selection.
- Fraud detection frequently requires analysing sequences of events rather than isolated records.
- Centralised configuration and reproducible random seeds make experimentation significantly easier.

One interesting observation was that some fraud patterns, such as bot activity and wash trading, became highly separable after feature engineering, while others, such as structuring behaviour, remained difficult to identify without examining user history over longer periods.

---

## Repository Structure

```
AnomX/
├── configs/
│   └── config.yaml              # Centralised configuration (seeds, paths, parameters)
├── data/
│   ├── raw/
│   │   └── events.csv           # Generated synthetic event dataset
│   └── processed/
│       └── features.csv         # Feature-engineered dataset
├── src/
│   ├── data_gen/
│   │   └── generate_events.py   # Synthetic event data generator
│   ├── features/
│   │   └── feature_engineering.py  # Feature pipeline
│   └── utils/
│       └── logger.py            # Shared logging utility
└── README.md
```
