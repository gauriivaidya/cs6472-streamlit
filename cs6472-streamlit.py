import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

import matplotlib as mpl
import matplotlib.pyplot as plt

# Reset style first (important)
mpl.rcdefaults()

# Set font globally
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

from plotnine import *
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel, wilcoxon, f_oneway

st.set_page_config(layout="wide")

st.title("Interactive Statistics and Model Explorer")

# ----------------------------
# Load dataset
# ----------------------------
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["species"] = iris.target
df["species"] = df["species"].map({
0:"setosa",
1:"versicolor",
2:"virginica"
})

features = df.columns[:-1]

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5= st.tabs(
[
"Dataset Overview",
"Visualizations",
"Statistical Tests",
"Model Comparison",
"Model Comparison with Statistical Testing"
]
)

# ----------------------------
# DATASET
# ----------------------------
with tab1:

    st.header("Dataset")

    st.dataframe(df.head())

    st.subheader("Data Types")

    st.write(df.dtypes)

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

# ----------------------------
# VISUALIZATION
# ----------------------------
with tab2:

    st.header("Visualization")

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Histogram","Boxplot","Violin","Scatter"]
    )

    x_var = st.selectbox("X Variable", features)

    if plot_type == "Scatter":
        y_var = st.selectbox("Y Variable", features)

    if plot_type == "Histogram":

        p = (
            ggplot(df, aes(x=x_var))
            + geom_histogram(bins=20)
        )

    elif plot_type == "Boxplot":

        p = (
            ggplot(df, aes(x="species", y=x_var, fill="species"))
            + geom_boxplot()
        )

    elif plot_type == "Violin":

        p = (
            ggplot(df, aes(x="species", y=x_var, fill="species"))
            + geom_violin()
        )

    elif plot_type == "Scatter":

        p = (
            ggplot(df, aes(x=x_var, y=y_var, color="species"))
            + geom_point()
        )

    st.pyplot(p.draw())

# ----------------------------
# STATISTICAL TESTS
# ----------------------------
with tab3:

    st.header("Statistical Tests")

    test = st.selectbox(
        "Select Test",
        ["T-test","ANOVA","Correlation"]
    )

    feature = st.selectbox(
        "Select Feature",
        features
    )

    if test == "T-test":

        g1 = df[df["species"]=="setosa"][feature]
        g2 = df[df["species"]=="versicolor"][feature]

        t,p = stats.ttest_ind(g1,g2)

        st.write("T statistic:",t)
        st.write("p-value:",p)

    if test == "ANOVA":

        g1 = df[df["species"]=="setosa"][feature]
        g2 = df[df["species"]=="versicolor"][feature]
        g3 = df[df["species"]=="virginica"][feature]

        f,p = stats.f_oneway(g1,g2,g3)

        st.write("F statistic:",f)
        st.write("p-value:",p)

    if test == "Correlation":

        var1 = st.selectbox("Variable 1", features, key="v1")
        var2 = st.selectbox("Variable 2", features, key="v2")

        corr,p = stats.pearsonr(df[var1],df[var2])

        st.write("Correlation:",corr)
        st.write("p-value:",p)

# # ----------------------------
# # MODELS
# # ----------------------------
# with tab4:

#     st.header("Model Comparison")

#     X = df[features]
#     y = iris.target

#     X_train,X_test,y_train,y_test = train_test_split(
#         X,y,test_size=0.3,random_state=42
#     )

#     models = {
#     "Logistic Regression": LogisticRegression(max_iter=200),
#     "KNN": KNeighborsClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier()
#     }

#     results = {}

#     for name,model in models.items():

#         if st.button(f"Run {name}"):

#             model.fit(X_train,y_train)

#             preds = model.predict(X_test)

#             acc = accuracy_score(y_test,preds)

#             st.write(name,"Accuracy:",acc)



from sklearn.model_selection import cross_val_score
from scipy.stats import ttest_rel, wilcoxon, f_oneway

with tab4:

    st.header("Model Comparison")

    X = df[features]
    y = iris.target

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    scores = {}

    if st.button("Run Models"):

        for name, model in models.items():

            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="accuracy"
            )

            scores[name] = cv_scores

            st.write(name)
            st.write("Fold Accuracies:", cv_scores)
            st.write("Mean Accuracy:", round(cv_scores.mean(),3))

        st.session_state["scores"] = scores

# -----------------------------------
# Statistical test selection
# -----------------------------------

if "scores" in st.session_state:

    scores = st.session_state["scores"]

    st.header("Statistical Test Selection")

    model_names = list(scores.keys())

    selected_models = st.multiselect(
        "Select Models to Compare",
        model_names
    )

    test = st.selectbox(
        "Select Statistical Test",
        ["Paired t-test", "Wilcoxon Test", "ANOVA"]
    )

    if st.button("Run Statistical Test"):

        if test == "Paired t-test":

            if len(selected_models) != 2:
                st.warning("Select exactly two models for a paired t-test")

            else:

                m1 = scores[selected_models[0]]
                m2 = scores[selected_models[1]]

                t, p = ttest_rel(m1, m2)

                st.write("t statistic:", t)
                st.write("p-value:", p)

        elif test == "Wilcoxon Test":

            if len(selected_models) != 2:
                st.warning("Select exactly two models")

            else:

                m1 = scores[selected_models[0]]
                m2 = scores[selected_models[1]]

                stat, p = wilcoxon(m1, m2)

                st.write("statistic:", stat)
                st.write("p-value:", p)

        elif test == "ANOVA":

            if len(selected_models) < 2:
                st.warning("Select two or more models")

            else:

                selected_scores = [
                    scores[m] for m in selected_models
                ]

                f, p = f_oneway(*selected_scores)

                st.write("F statistic:", f)
                st.write("p-value:", p)
