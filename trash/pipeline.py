import io
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from mlens.ensemble import SuperLearner
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Title
st.title("Diabetes Prediction ML Pipeline")

# **1. Upload Data**
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

# **2. Exploratory Data Analysis (EDA)**
st.header("Step 2: Exploratory Data Analysis (EDA)")

if st.checkbox("Show Descriptive Statistics"):
    st.write("### Descriptive Statistics")
    st.write(df.describe())

    buffer = io.StringIO()  # Capture output of df.info()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

if st.checkbox("Show Data Visualization"):
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot by Outcome")
    pairplot_fig = sns.pairplot(df, hue="Outcome", palette="husl")
    st.pyplot(pairplot_fig)

    
    # **Outliers Investigation**
    st.subheader("Outliers Investigation")
    if st.checkbox("Investigate Outliers (Single Feature)"):
        feature = st.selectbox("Select feature for outlier detection", df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(df[feature], ax=ax)
        st.pyplot(fig)
    
    if st.checkbox("Investigate Outliers (Pairs)"):
        feature_x = st.selectbox("Select feature X", df.columns)
        feature_y = st.selectbox("Select feature Y", df.columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[feature_x], y=df[feature_y], ax=ax)
        st.pyplot(fig)

    # **3. Data Preprocessing**
    st.header("Step 3: Data Preprocessing")
    
    # Feature Scaling Options
    st.write("### Feature Scaling")
    scaler_option = st.radio("Choose a scaling method", ["Standard Scaler", "MinMax Scaler"])
    
    # Splitting the dataset
    X = df.drop(columns=['Outcome'])  # Replace 'Outcome' with your target column
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if scaler_option == "Standard Scaler":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Data Scaled using", scaler_option)
    
    # **4. Feature Engineering**
    st.header("Step 4: Feature Engineering")
    st.write("### Correlation Analysis")
    st.write("Select features based on correlation and importance.")
    
    # **5. Model Training**
    st.header("Step 5: Model Training - Ensemble Methods")

    # Stacking Classifier
    def get_models():
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier

        models = {
            "lr": LogisticRegression(),
            "knn": KNeighborsClassifier(),
            "svc": SVC(probability=True),
            "rf": RandomForestClassifier()
        }
        return models

    if st.button("Train Model"):
        base_learners = get_models()
        meta_learner = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001, random_state=42)
        sl = SuperLearner(folds=10, random_state=42, backend="multiprocessing")
        sl.add(list(base_learners.values()), proba=True)
        sl.add_meta(meta_learner, proba=True)

        # Train the ensemble
        sl.fit(X_train_scaled, y_train)
        
        # **6. Evaluation Metrics**
        st.header("Step 6: Evaluation Metrics")

        # Prediction
        y_pred_proba = sl.predict_proba(X_test_scaled)[:, 1]
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_proba]

        # Display metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"**ROC-AUC Score**: {roc_auc:.3f}")
        st.write(f"**Accuracy**: {accuracy:.3f}")
        st.write(f"**Precision**: {precision:.3f}")
        st.write(f"**Recall**: {recall:.3f}")
        st.write(f"**F1 Score**: {f1:.3f}")

        st.write("### Confusion Matrix")
        st.write(conf_matrix)

