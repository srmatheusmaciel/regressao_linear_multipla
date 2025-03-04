# Cholesterol Level Prediction using Multiple Linear Regression

## Project Overview

This project develops a machine learning model to predict total cholesterol levels using multiple linear regression, leveraging various demographic and lifestyle factors.

## Project Structure
```
regressao_linear_multipla/
│
├── .gradio/                # Gradio app configuration
├── datasets/               # Data storage
│   └── dataset_colesterol.csv
├── app_gradio_colesterol.ipynb   # Gradio web app notebook
├── modelo_colesterol.ipynb       # Model training notebook
└── README.md               # Project documentation
```

## Prerequisites

### Environment Setup
- Python 3.11+


## Libraries Used

### Scientific and Data Analysis
- **Pandas**: Data manipulation and analysis  
- **NumPy**: Numerical computation and mathematical operations  
- **SciPy**: Scientific computing and statistical analysis

### Machine Learning
- **Scikit-learn**:  
  - Machine learning algorithms  
  - Data preparation  
  - Evaluation metrics  
  - Dataset splitting

### Visualization
- **Matplotlib**: Creation of static plots  
- **Seaborn**: Advanced statistical visualizations

### Interface
- **Gradio**:  
  - Creation of interactive web interfaces  
  - Quick model demonstration

### Statistics
- **Pingouin**:  
  - Statistical analysis  
  - Hypothesis testing  
  - Advanced correlations

### Development Environment
- **IPykernel**:  
  - Support for Jupyter Notebooks  
  - Interactive Python code execution

## Setup

### Dependency Installation
```bash
pipenv install scikit-learn scipy pandas matplotlib seaborn ipykernel gradio pingouin numpy
```

## Data Features
The `dataset_colesterol.csv` includes the following columns:
- Grupo Sanguíneo (Blood Group)
- Fumante (Smoker Status)
- Nível de Atividade (Activity Level)
- Idade (Age)
- Peso (Weight)
- Altura (Height)
- Colesterol (Cholesterol Level)

## Model Development
- **Technique**: Multiple Linear Regression
- **Goal**: Predict total cholesterol levels
  
## Project Stages

1. **EDA (Exploratory Data Analysis)**
- Initial investigation of the dataset  
- Identification of patterns and insights

2. **Dataset Preparation**
- Data cleaning  
- Handling missing values  
- Feature normalization and standardization

3. **Model Training**
- Splitting data into training and testing sets  
- Implementation of Multiple Linear Regression

4. **Model Validation**
- Performance metric evaluation  
- Error and accuracy analysis

5. **Value Prediction**
- Generating predictions for new data  
- Interpreting the results

6. **Save Model**
- Persistence of the trained model

7. **Deployment with Gradio App**
- Creating an interactive interface for predictions

## Visualization and Deployment
- Interactive web interface using Gradio
- Visualizations of model performance and feature importance

## Model Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Usage
1. Activate virtual environment
2. Run Jupyter notebook for model training
3. Launch Gradio app for predictions

## Future Improvements
- Experiment with advanced regression techniques
- Collect more diverse data
- Implement cross-validation
- Enhance feature engineering

---

## Author

### Matheus Maciel
You can find me on [LinkedIn](https://www.linkedin.com/srmatheusmaciel) or [GitHub](https://github.com/srmatheusmaciel).








