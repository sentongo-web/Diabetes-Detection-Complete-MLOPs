# Nakaseke NCD-AI: Hypertension Risk Screener

**A world-class, end-to-end machine learning system for community hypertension screening built from real patient data collected at Nakaseke Hospital, Uganda.**

---

## What This Project Does and Why It Matters

High blood pressure (hypertension) is the single biggest cause of heart attacks and strokes in sub-Saharan Africa. The tragedy is that it has no symptoms. Most people do not know they have it until they suffer a life-changing event. At Nakaseke Hospital in Uganda, nurses and community health workers screen thousands of patients each year, but access to clinical equipment is limited and many patients in rural areas never get a blood pressure measurement at all.

This project builds an artificial intelligence model that can estimate a person's hypertension risk from a short questionnaire alone — no blood test, no cuff, no equipment needed. A community health worker with a basic smartphone could use this tool to identify who most urgently needs a clinical assessment, allowing scarce healthcare resources to be directed where they are needed most.

---

## Why Hypertension, Not Diabetes?

The project repository is named "Diabetes-Detection" but the dataset tells a different story. When we opened the raw data files and analysed them rigorously, the numbers were clear:

**The raw data from Nakaseke Hospital contains:**
- Only 54 confirmed diabetic patients in 3,471 total (1.6% positive rate) from blood glucose measurements
- 1,248 patients with hypertension confirmed by blood pressure readings (36% positive rate)

A machine learning model needs both positive and negative examples to learn from. With only 1.6% positive cases for diabetes, even a model that always predicts "no diabetes" would be 98.4% accurate — but completely useless clinically. This is called the **class imbalance problem**.

Hypertension at 36% prevalence is near-perfect for machine learning. The model has enough examples of both hypertensive and normotensive patients to learn meaningful patterns. More importantly, the Nakaseke study was designed as a Non-Communicable Diseases (NCD) screening study, and hypertension is the primary NCD in Uganda.

**The diabetes labels from the supplementary CSV are kept in the repository for secondary analysis.** The production model targets hypertension.

---

## The Dataset: Nakaseke Hospital NCD Survey

The data was collected through a community-based NCD screening programme at Nakaseke Hospital, Uganda. Community Health Workers (CHWs) visited households in surrounding villages and recorded:

- **Demographics**: age, sex, education level, marital status, employment, household size
- **Lifestyle**: smoking history, alcohol consumption (type, frequency, quantity)
- **Diet**: fruit and vegetable intake, salt use habits, processed food frequency, cooking oil type
- **Environment**: main cooking fuel (firewood/charcoal exposure linked to cardiovascular risk)
- **Physical activity**: vigorous and moderate exercise frequency
- **Body measurements**: height, weight, waist circumference, hip circumference
- **Clinical measurements**: three blood pressure readings per patient, fasting and random blood glucose, full urinalysis

The raw data is stored in Stata (.dta) format from REDCap/SurveyCTO electronic data collection. A supplementary CSV provides pre-extracted and cleaned diabetes-related variables.

**Dataset summary after cleaning:**

| Property | Value |
|---|---|
| Total patients | 3,471 |
| Hypertensive patients | 955 (27.5%) |
| Normotensive patients | 2,516 (72.5%) |
| Predictor features used | 36 |
| Missing data after imputation | 0% |
| Data source | Nakaseke Hospital, Uganda |

---

## Project Structure

Every file has a clear purpose. Here is what lives where and why:

```
Nakaseke NCD-AI/
|
|-- data/
|   |-- raw/                     Original data files (never modified)
|   |   |-- RIF_final.dta        3,471-patient Stata research file (primary)
|   |   |-- diabetes_dataset.csv Pre-extracted CSV (secondary reference)
|   |-- processed/
|       |-- nakaseke_clean.csv   Cleaned, validated, model-ready data
|
|-- src/                         All Python source code
|   |-- config.py                Loads config/config.yaml as Python objects
|   |-- data/
|   |   |-- loader.py            Reads raw .dta and .csv files safely
|   |   |-- cleaner.py           Cleans, validates, encodes, imputes data
|   |   |-- features.py          Engineers new clinical composite features
|   |-- models/
|   |   |-- trainer.py           Trains, tunes, evaluates 6 models + SHAP
|   |-- visualization/
|       |-- plots.py             Creates all EDA and evaluation plots
|
|-- app/                         Flask web application
|   |-- app.py                   Routes, prediction logic, risk interpretation
|   |-- templates/
|   |   |-- base.html            Shared navigation, hero, footer
|   |   |-- index.html           Patient questionnaire form
|   |   |-- result.html          Risk score, gauge, recommendations
|   |-- static/
|       |-- css/style.css        Medical-grade responsive design
|       |-- js/main.js           Live BMI calculator, form validation
|
|-- models/                      Trained model artefacts
|   |-- best_model.pkl           Serialised production pipeline
|   |-- feature_names.csv        Ordered list of 36 model inputs
|   |-- shap_values.npy          SHAP explanation values (test set)
|   |-- figures/                 All generated plots (EDA, ROC, SHAP)
|
|-- config/
|   |-- config.yaml              Central configuration (all magic numbers)
|
|-- train.py                     One-command pipeline runner
|-- Dockerfile                   Multi-stage Docker build
|-- docker-compose.yml           App + MLflow tracking stack
|-- requirements.txt             All Python package dependencies
|-- Makefile                     Developer shortcuts
```

---

## How to Run This Project

### Option 1: Run Locally (Recommended for Development)

**Step 1 — Clone the repository**
```bash
git clone https://github.com/sentongo-web/Diabetes-Detection-Complete-MLOPs.git
cd Diabetes-Detection-Complete-MLOPs
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Place the raw data files**

Copy `RIF_final.dta` and `diabetes_dataset.csv` into `data/raw/`. These files contain patient data and are not committed to GitHub.

**Step 4 — Train the model**

This runs the entire pipeline: loads data, cleans it, engineers features, trains 6 models, runs Optuna hyperparameter search, evaluates on the test set, computes SHAP values, and saves the best model.
```bash
python train.py
```
Training takes about 5 minutes on a modern laptop. Progress is printed to the terminal and saved to `training.log`.

**Step 5 — Start the web application**
```bash
python app/app.py
```
Open your browser at `http://localhost:5000`.

### Option 2: Run with Docker

**Build and run:**
```bash
docker build -t nakaseke-ncd-ai .
docker run -p 5000:5000 nakaseke-ncd-ai
```

**Full stack with MLflow tracking UI:**
```bash
docker compose up --build
```
- Web application: `http://localhost:5000`
- MLflow tracking:  `http://localhost:5001`

### Option 3: Make Shortcuts

```bash
make install      # Install dependencies
make train        # Train the model
make run          # Start the Flask app
make docker-build # Build Docker image
make mlflow       # Start the MLflow UI
make test         # Run the test suite
```

---

## The Machine Learning Pipeline — Step by Step

This section explains every step in the pipeline, how the code works, and the decisions made along the way.

### Step 1: Loading the Data (`src/data/loader.py`)

The `load_rif()` function uses `pandas.read_stata()` to read the Stata binary format, preserving all variable labels as category types. We keep loading completely separate from cleaning — any analyst can import just the loader and see the original data exactly as it was collected, before any transformation. This separation is fundamental to reproducible research.

```python
# Reading the Stata file preserves Stata's category labels
df = pd.read_stata("data/raw/RIF_final.dta", convert_categoricals=True)
# Result: 3,471 rows × 287 columns, categories like "YES"/"NO", "MALE"/"FEMALE"
```

### Step 2: Data Cleaning (`src/data/cleaner.py`)

This is the most complex and most important module. Raw clinical data collected in the field is messy in ways that general-purpose cleaning tools cannot handle. Every decision is documented below.

**Building the target variable from blood pressure:**

We do NOT use the self-reported disease column. Self-report is subject to recall bias, misdiagnosis, and reporting bias. Instead, we compute hypertension from the actual blood pressure measurements using the WHO/JNC-8 clinical definition:

> Hypertensive = systolic BP >= 140 mmHg OR diastolic BP >= 90 mmHg

Each patient had three readings taken. We average them to reduce white-coat effect and measurement noise. Readings outside the physiological range are discarded before averaging — not clamped — because clamping would introduce false precision.

```python
# Average three readings after discarding impossible values
sbp_mean = mean of readings where 70 <= systolic <= 250
dbp_mean = mean of readings where 40 <= diastolic <= 140
label = 1 if sbp_mean >= 140 OR dbp_mean >= 90
```

**Cleaning anthropometric measurements:**

The height column contains values in both centimetres (e.g. 160) and millimetres (e.g. 1600). We detect the millimetre entries by scale (>250) and divide by 10 to convert to centimetres. All measurements are validated against physiological bounds.

**Encoding categorical variables:**

The Stata file uses mixed-language string labels. We map these carefully using domain knowledge:

| Raw Stata value | Encoded as | Why |
|---|---|---|
| "FEMALE" / "MALE" | 1 / 0 | Standard clinical encoding |
| Education levels | 0-6 ordinal | Higher education = lower risk; order matters |
| "Married" / "Cohabiting" | 1; others = 0 | Social support is a protective factor |
| "Never/Rarely/Sometimes/Often/Always" | 0-4 ordinal | Likert scale preserves frequency gradient |
| Cooking oil type | 0=solid/other, 1=vegetable, 2=olive | Reflects cardiovascular health profile |
| Firewood/Charcoal | 1 | Biomass smoke is a cardiovascular risk factor |

**Imputing missing values:**

After cleaning, some values remain missing. We use median for continuous variables (robust to skewed distributions) and mode for categorical ones. Critically, imputation statistics are computed on the training fold only, never seeing the test data.

### Step 3: Feature Engineering (`src/data/features.py`)

Raw features are good. Engineered features that encode clinical knowledge are better. We create 11 additional features:

**BMI category (`bmi_cat`):** The WHO classification (underweight=0, normal=1, overweight=2, obese=3) captures the non-linear relationship between weight and cardiovascular risk. Being "overweight" is qualitatively different from being "obese."

**Age group (`age_group`):** Hypertension risk increases sharply with age, not linearly. Encoding age bands (under 30, 30-44, 45-59, 60+) lets models capture this threshold behaviour.

**Waist risk flag (`waist_risk`):** WHO defines abdominal obesity as waist > 88 cm for women and > 102 cm for men. Abdominal fat is more metabolically harmful than overall obesity. This flag is a direct clinical risk indicator.

**WHR risk flag (`whr_risk`):** Waist-to-hip ratio captures body fat distribution. High WHR means more central fat, a stronger cardiovascular predictor than BMI alone. Women: WHR > 0.85. Men: WHR > 0.90.

**Lifestyle score (`lifestyle_score`):** A composite count (0-6) of concurrent risk factors: smoking, heavy drinking (>14 units/week), obesity (BMI>=25), low fruit/vegetable intake, excess salt, and physical inactivity. This gives the model a holistic signal beyond individual behaviours.

**Metabolic burden (`metabolic_burden`):** A count (0-3) of metabolic syndrome components: obesity, abdominal obesity, high WHR. A patient with all three is at far higher risk than a patient with just one.

**Interaction terms (`age_bmi`, `age_whr`, `bmi_salt`):** Products of standardised feature pairs. These let linear models detect interactions they could not otherwise capture — for example, that the combination of being old AND obese is more dangerous than either alone.

**Polynomial terms (`age_sq`, `bmi_sq`):** Squared versions of age and BMI, because the relationship between these variables and blood pressure accelerates at higher values.

### Step 4: Exploratory Data Analysis (`src/visualization/plots.py`)

Before modelling, we generate a complete diagnostic picture:

- **Target distribution** — confirming 27.5% / 72.5% split, viable for learning
- **Age by hypertension status** — hypertensive patients are on average older
- **BMI by status** — hypertensive patients have higher BMI, especially past BMI 30
- **Correlation matrix** — identifying multicollinear features (e.g. waist_cm and hip_cm correlate, whr and waist_risk correlate, as expected clinically)
- **Missing data profile** — showing which raw columns had the most gaps before imputation

All plots are saved to `models/figures/` as PNG files for inspection and reporting.

### Step 5: Model Training and Selection (`src/models/trainer.py`)

We follow a rigorous, unbiased evaluation protocol with four key safeguards against overfitting.

**Safeguard 1 — Three-way stratified split:**

70% training, 10% validation, 20% held-out test. The test set is locked away completely and touched only once, at the very end. Stratification ensures the 27.5% positive rate is preserved in all three splits. Using the test set to select models or tune hyperparameters is data leakage that inflates reported performance.

**Safeguard 2 — SMOTE applied only inside training folds:**

SMOTE creates synthetic minority-class examples by interpolating between real ones. It is applied inside the sklearn Pipeline, which means it only ever runs on training data — never on validation or test data. This is the correct way to use oversampling.

**Safeguard 3 — 5-fold cross-validation for benchmarking:**

Each candidate model is trained and evaluated five times on different partitions of the training data. The mean AUC across folds is a more reliable performance estimate than a single evaluation.

**Safeguard 4 — Held-out test set for final reporting:**

After selecting the winner through validation AUC and Optuna tuning, the model is retrained on the combined train+validation set and evaluated once on the test set. This single evaluation is the honest performance estimate reported in the results table.

**The six candidate models:**

| Model | Strengths in this context |
|---|---|
| Logistic Regression | Interpretable; handles linear relationships well; winner here |
| Random Forest | Handles non-linearity; robust to outliers; built-in feature importance |
| Gradient Boosting | Powerful ensemble; good for tabular data |
| XGBoost | State-of-the-art for structured data; Optuna-tuned |
| LightGBM | Faster than XGBoost for large datasets; Optuna-tuned |
| K-Nearest Neighbours | Non-parametric baseline; distance-based reasoning |

**Optuna hyperparameter optimisation:**

The top two models by validation AUC are passed to Optuna, which uses Bayesian optimisation (Tree-structured Parzen Estimator) to find better hyperparameters in 60 trials. The objective function for each trial is 5-fold CV AUC on the training set. This is smarter and more efficient than grid search.

**SHAP explanations:**

After selecting the winner, SHAP (SHapley Additive exPlanations) values are computed for every test set prediction. SHAP tells us exactly how much each feature pushed the prediction higher or lower for each individual patient. This is the gold standard for clinical AI explainability because it is grounded in game theory and satisfies mathematical desiderata (efficiency, symmetry, linearity).

**MLflow experiment tracking:**

Every training run is tracked automatically: the model type, all hyperparameters, all metrics, and the trained model artefact. You can open the MLflow UI (`make mlflow`) and compare all six models side by side, reproduce any run, and register the best model in the model registry.

### Step 6: Flask Web Application (`app/app.py`)

The web application serves a questionnaire that mirrors the training data collection process. When a patient submits their answers:

1. The form data is validated and sanitised server-side (all values clamped to physiological ranges; no SQL or shell injection surface exists)
2. The 36-feature vector is constructed using the exact same engineering logic as training
3. The saved sklearn Pipeline applies its imputer then the trained classifier
4. The output probability is interpreted into a risk band: Low (<25%), Moderate (25-50%), High (50-70%), Very High (>70%)
5. Personalised recommendations are generated based on which specific risk factors are present
6. The result page displays a gauge chart built with inline SVG, the risk band with colour coding, a clinical interpretation, lifestyle tips, and WHO diagnostic thresholds for reference

The app never stores any patient data. All processing happens in memory and is discarded after the response is sent.

---

## Model Performance

| Metric | Training CV | Validation | Test (held-out) |
|---|---|---|---|
| ROC-AUC | 0.659 | 0.689 | 0.639 |
| F1 Score | — | — | 0.447 |
| Recall | — | — | 0.571 |
| Precision | — | — | 0.367 |
| Accuracy | — | — | 0.612 |

**Why is the AUC 0.64 and not 0.90?**

This is a question worth answering honestly. The model predicts hypertension from questionnaire and anthropometric data alone — no blood pressure readings, no ECG, no laboratory tests. This is intentional. The whole point is to screen patients who have not yet had a clinical measurement.

Published literature for community-based hypertension risk models using questionnaire data in low- and middle-income countries consistently reports AUC values between 0.60 and 0.72. Our model at 0.64 sits squarely in this range and is consistent with what is clinically achievable from this type of data.

A model that used blood pressure readings as input features would achieve AUC > 0.99, because high blood pressure IS hypertension. But that model would be useless as a screening tool — you would already have the diagnosis before running the model.

The clinical value here is the ability to identify 57% of hypertensive patients (recall = 0.571) before they have a blood pressure measurement, using only information a health worker can collect without any equipment.

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Data processing | pandas, numpy | Standard, well-tested, fast on tabular data |
| Stata file reading | pyreadstat | Native .dta support preserving variable labels |
| Machine learning | scikit-learn, XGBoost, LightGBM | Industry standards; complementary strengths |
| Class balancing | imbalanced-learn SMOTE | State-of-the-art for minority oversampling |
| Hyperparameter tuning | Optuna | Bayesian search; far more efficient than grid search |
| Experiment tracking | MLflow | Open-source; integrates with all major frameworks |
| Explainability | SHAP | Theoretically grounded; accepted in clinical AI |
| Visualisation | matplotlib, seaborn | Publication-quality figures |
| Web server | Flask + Gunicorn | Lightweight; easy to containerise |
| Containerisation | Docker multi-stage build | Reproducible deployment anywhere |
| Orchestration | Docker Compose | Multi-service stack for local and cloud deployment |
| Configuration | YAML + Python SimpleNamespace | Central config; no magic numbers in code |

---

## Ethical Considerations

**Privacy:** Patient identifiers (name, phone number, date of birth, GPS coordinates) are excluded from all model inputs. The cleaned dataset contains no personal identifiers.

**Transparency:** SHAP values make the model's reasoning visible. A clinician can see exactly which factors contributed to a high-risk prediction and can question and verify the result rather than blindly accepting it.

**Calibration:** The model outputs a probability, not just a binary label. Clinical users can apply their own judgment about risk thresholds rather than having a fixed cutoff imposed.

**Scope limitation:** The tool explicitly states on every result page that it is a screening tool, not a diagnosis. It recommends clinical follow-up for all moderate-to-high risk results.

**Non-storage:** The web application never stores patient responses. Processing is in-memory only.

---

## Reproducing the Results

All random operations use a fixed seed (42) set in `config/config.yaml`. Given the same data, the training script produces the same model on every run.

```bash
pip install -r requirements.txt
python train.py
# Check training.log and models/figures/ for detailed outputs
```

The MLflow tracking server records every run so you can compare training runs side by side and reproduce any result.

---

## Contributing

This project is open for research collaboration. Particularly valuable contributions:

- Additional patient data from Nakaseke or other Ugandan health facilities
- Independent cohort validation
- Translation of the web interface into Luganda or other local languages
- Integration with mobile health platforms such as OpenMRS or DHIS2

Please open an issue or pull request on GitHub.

---

## Citation

If you use this work in research, please cite:

```
Sentongo, P. (2024). Nakaseke NCD-AI: End-to-end Machine Learning for Community
Hypertension Screening in Uganda. GitHub.
https://github.com/sentongo-web/Diabetes-Detection-Complete-MLOPs
```

---

## License

MIT License. The raw patient data files are property of Nakaseke Hospital Research Unit and are used with permission for research purposes. They are not included in this repository.

---

*Built with real data from real patients at Nakaseke Hospital, Uganda.*
