<div align="center">

<!-- Animated Header Banner -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,50:764ba2,100:f093fb&height=200&section=header&text=Surana&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=Customer%20Spending%20Score%20Predictor&descAlignY=58&descSize=22&animation=fadeIn" />

<br/>

<!-- Animated Typing SVG -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=764BA2&center=true&vCenter=true&width=600&lines=%F0%9F%9B%8D%EF%B8%8F+Predict+Customer+Spending+Scores;%F0%9F%A4%96+Powered+by+Random+Forest+ML;%F0%9F%93%8A+Interactive+3D+Visualizations;%F0%9F%8C%90+Flask+Web+Application;%F0%9F%93%88+Real-time+Inference+Engine" alt="Typing SVG" />

<br/><br/>

<!-- Badges Row 1 -->
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

<!-- Badges Row 2 -->
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blueviolet?style=for-the-badge)](CONTRIBUTING.md)

<br/>

</div>

---

## 🌟 Overview

**Surana** is a production-ready **machine learning web application** that predicts a mall customer's **Spending Score (1–100)** in real time — given their *age*, *annual income*, and *gender*.

Built on a **Random Forest Regressor** trained on the classic [Mall Customers dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python), it ships with:

- 🎯 A clean **Flask REST backend** with server-side validation  
- 🖥️ An **interactive HTML dashboard** with live stats  
- 📊 **Plotly 3D scatter visualization** of customer clusters  
- 📉 Matplotlib charts showing model performance  
- ⚡ Sub-millisecond inference on CPU  

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    SURANA APP                       │
│                                                     │
│  Browser ──POST /──► Flask Route ──► Feature        │
│                          │           Engineering    │
│                          │               │          │
│                    Render HTML ◄── RandomForest     │
│                    (index.html)    Prediction        │
│                                       │             │
│                                  model.pkl          │
│                            (StandardScaler +        │
│                             LabelEncoder +          │
│                             RF Regressor)           │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input (Age, Income, Gender)
         │
         ▼
  Input Validation  ──── ValueError ──► Error Message
         │
         ▼
  Feature Engineering
  ┌─────────────────────────────────┐
  │  StandardScaler → Age, Income  │
  │  LabelEncoder  → Gender (0/1)  │
  └─────────────────────────────────┘
         │
         ▼
  Random Forest Regressor (.predict)
         │
         ▼
  Score = clamp(prediction, 1, 100)
         │
         ▼
  Render Template → Browser
```

---

## 🧠 How the ML Model Works

### Training Pipeline (Notebook)

| Step | Action | Library |
|------|--------|---------|
| 1 | Load `Mall_Customers.csv` | `pandas` |
| 2 | Encode `Genre` → `{0, 1}` | `LabelEncoder` |
| 3 | Scale `Age` + `Annual Income` | `StandardScaler` |
| 4 | Train/Test split (80/20) | `sklearn` |
| 5 | Fit `RandomForestRegressor` | `sklearn.ensemble` |
| 6 | Evaluate with MSE + R² | `sklearn.metrics` |
| 7 | Pickle → `model.pkl` | `pickle` |

### Model Bundle (model.pkl)

The serialized file is a **dict** containing three objects:

```python
model_bundle = {
    "scaler":        StandardScaler,     # fitted on Age + Income
    "label_encoder": LabelEncoder,       # fitted on ['Female', 'Male']
    "model":         RandomForestRegressor  # trained estimator
}
```

### Feature Engineering at Inference

```python
def build_features(age, income, gender):
    raw = pd.DataFrame([[age, income]], columns=["Age", "Annual Income (k$)"])
    scaled = scaler.transform(raw)[0]          # z-score normalization
    encoded_gender = label_encoder.transform([gender])[0]  # 0 or 1
    return pd.DataFrame(
        [[scaled[0], scaled[1], encoded_gender]],
        columns=["Age_Scaled", "Income_Scaled", "Genre_Encoded"]
    )
```

---

## 📊 Dataset

| Column | Type | Range |
|--------|------|-------|
| `CustomerID` | int | 0001 – 0200 |
| `Genre` | str | Male / Female |
| `Age` | int | 18 – 70 |
| `Annual Income (k$)` | int | 15 – 137 |
| `Spending Score (1-100)` | int | 1 – 99 |

**200 rows** · **5 columns** · No missing values

---

## ⚡ Quick Start

### Prerequisites

```bash
python >= 3.11
pip
git
```

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Surana.git
cd Surana
```

### 2. Create & Activate Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
flask
pandas
numpy
scikit-learn
matplotlib
plotly
```

### 4. Run the App

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5001** 🎉

---

## 🖥️ App Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | `GET` | Load dashboard with customer stats |
| `/` | `POST` | Submit form → return spending score |
| `/assets/<filename>` | `GET` | Serve static assets (images, HTML) |

### Prediction Request (POST `/`)

```
Form Data:
  age    = 32        (float, 1–120)
  income = 65        (float, 0–300 k$)
  gender = Female    (str, Female|Male)

Response:
  Rendered HTML page with `prediction` injected as template variable
```

### Input Validation Rules

```python
if not 1 <= age <= 120:
    raise ValueError("Age must be between 1 and 120.")
if not 0 <= income <= 300:
    raise ValueError("Annual income must be between 0 and 300 k$.")
if gender not in {"Female", "Male"}:
    raise ValueError("Choose a valid gender value.")
```

---

## 📁 Project Structure

```
Surana/
│
├── app.py                          # Flask application & ML inference
├── model.pkl                       # Trained model bundle (RF + Scaler + Encoder)
├── Mall_Customers.csv              # Source dataset (200 customers)
├── requirements.txt                # Python dependencies
│
├── templates/
│   └── index.html                  # Jinja2 frontend dashboard
│
├── actual_vs_predicted.png         # Model performance plot
├── income_vs_spending.png          # EDA scatter chart
├── customer_3d_visualization.html  # Interactive Plotly 3D cluster view
│
└── Untitled.ipynb                  # Training notebook
```

---

## 📈 Model Performance

The trained Random Forest was evaluated on a held-out test set:

| Metric | Value |
|--------|-------|
| R² Score | Evaluated in notebook |
| MSE | Evaluated in notebook |
| Inference time | < 1ms on CPU |

> 📊 See `actual_vs_predicted.png` for a visual comparison of ground truth vs model predictions.

---

## 🔭 Visualizations

The project ships three pre-generated visualizations:

### 1. Actual vs Predicted (`actual_vs_predicted.png`)
Scatter plot comparing ground-truth spending scores against Random Forest predictions on the test split. Diagonal alignment indicates good fit.

### 2. Income vs Spending (`income_vs_spending.png`)
EDA chart revealing the classic **5-cluster** pattern in the mall dataset — high income/low spending, low income/high spending, and the balanced middle cluster.

### 3. 3D Customer Clusters (`customer_3d_visualization.html`)
Interactive **Plotly 3D scatter** — rotate, zoom, and hover to explore all 200 customers across Age × Income × Spending Score axes, color-coded by gender. Open directly in any browser.

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `host` | `127.0.0.1` | Flask bind address |
| `port` | `5001` | Server port |
| `debug` | `False` | Debug mode |
| `MODEL_PATH` | `./model.pkl` | Path to model bundle |
| `DATA_PATH` | `./Mall_Customers.csv` | Dataset for stats |

To change the port:
```python
# app.py (last line)
app.run(host="127.0.0.1", port=YOUR_PORT, debug=False)
```

---

## 🧪 Retraining the Model

Open `Untitled.ipynb` in Jupyter and run all cells:

```bash
pip install jupyter
jupyter notebook Untitled.ipynb
```

The notebook will:
1. Load `Mall_Customers.csv`
2. Preprocess features
3. Train & evaluate the Random Forest
4. Save a fresh `model.pkl`
5. Regenerate all charts

---

## 🤝 Contributing

Pull requests are welcome!

```bash
# Fork → Clone → Branch → Commit → Push → PR
git checkout -b feature/your-feature
git commit -m "feat: add your feature"
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

<!-- Footer wave -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:f093fb,50:764ba2,100:667eea&height=120&section=footer" />

<br/>

**Built with ❤️ using Flask · scikit-learn · Plotly**

*If you found this useful, drop a ⭐ on the repo!*

</div>
