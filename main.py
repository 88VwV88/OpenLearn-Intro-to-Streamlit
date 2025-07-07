import streamlit as st
from sklearn.datasets import load_iris

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
)

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

sns.set_style("darkgrid")

X, y = load_iris(as_frame=True, return_X_y=True)
data = X.assign(species=y)


def dataset_visualization():
    st.title("Dataset Visualization")

    # iris dataset visualizations:

    st.divider()
    # st.header and st.subheader can be used to create separate blocks
    st.header("Iris dataset visualizations")

    # st.code can be used to print the output, using syntax highlighting
    # use st.text for non code text output.
    st.code(X.head())
    st.code((X.shape, y.shape))

    # creating 2 column code blocks using st.column
    st.subheader("Dataset NULL counts and dtypes")
    null_col, dtype_col = st.columns(2)
    with null_col:
        st.code(X.isna().sum())
    with dtype_col:
        st.code(X.dtypes)

    # dataset feature visualizations using seaborn and st.pyplot

    st.divider()
    # pairplot for each feature pairs
    fig = sns.pairplot(data, hue="species")
    st.pyplot(fig)

    # plotting the boxplot for each feature
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_figwidth(15)
    for i in range(4):
        sns.boxplot(x=X.iloc[..., i], ax=ax[i])
    st.pyplot(fig)

    # plotting the histplot for each feature
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_figwidth(15)
    for i in range(4):
        sns.histplot(x=X.iloc[..., i], kde=True, ax=ax[i])
    st.pyplot(fig)

    st.divider()

    st.markdown("""### Observations from visualization:
    * use robust scaler for sepal width (cm) due to presence of outliers
    * use standard scaler for sepal length (cm) due to normal distribution
    * use min-max scaler for the other two non-normal distributions
    * features are non linear so use linear and kernel PCA
    use the LogisticRegressionCV model for multi-class classification.""")


use_pca = st.sidebar.checkbox("Use PCA decomposition", value=False)
preprocessor = Pipeline(
    [
        (
            "scale",
            ColumnTransformer(
                [
                    ("standard", StandardScaler(), ["sepal length (cm)"]),
                    ("robust", RobustScaler(), ["sepal width (cm)"]),
                ],
                remainder=MinMaxScaler(),
            ),
        ),
        (
            "decompose",
            FeatureUnion([("linear_pca", PCA()), ("kernel_pca", KernelPCA())])
            if use_pca
            else FunctionTransformer(lambda X: X),
        ),
    ]
)


def logistic_regression():
    st.title("Logistic Regression CV")

    st.subheader("Model parameters")
    cols = st.columns(2)

    with cols[0]:
        use_cv = st.checkbox("Use Cross Validation model", value=True)

    if use_cv:
        with cols[1]:
            num_cv_folds = st.number_input("#cv_folds", value=3, step=1, format="%d")
            st.text("CV fold lambda values")
            cv_Cs = [0.5, 1.5, 2.5]
            for i, C in enumerate(cv_Cs):
                cv_Cs[i] = st.number_input(f"[{i}]", value=C)

    st.divider()

    log_model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegressionCV(Cs=cv_Cs, cv=num_cv_folds, random_state=42)
                if use_cv
                else LogisticRegression(),
            ),
        ]
    )

    st.code(log_model)

    # train the model on various splits
    sss = StratifiedShuffleSplit(random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]

        log_model.fit(X_train, y_train)
        y_pred = log_model.predict(X_test)

        st.write(
            f"[`{i + 1:2}`] **LOSS**: `{mean_squared_error(y_test, y_pred):.3f}` **ACCURACY**: `{accuracy_score(y_test, y_pred):.3f}`"
        )

    st.divider()
    st.subheader("Classification report")

    y_pred = log_model.predict(X)
    st.code(classification_report(y, y_pred))


def random_forest():
    st.title("Random Forest Classifier")

    st.subheader("Model parameters")
    criterion = (
        st.pills("Optimizer Criterion", ["gini", "entropy", "log_loss"]) or "gini"
    )
    max_depth = st.number_input("Maximum Forest Depth", value=15, step=1) or None
    min_samples_split = st.number_input("Minimum samples for splitting", value=2)
    max_features = st.selectbox("Maximum features", ["sqrt", "log2"])

    st.divider()

    rf_model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    criterion=criterion,
                    min_samples_split=min_samples_split,
                    max_depth=max_depth,
                    max_features=max_features,
                ),
            ),
        ]
    )

    st.code(rf_model)

    sss = StratifiedShuffleSplit(random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]

        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        st.write(
            f"[`{i + 1:2}`] **LOSS**: `{mean_squared_error(y_test, y_pred):.3f}` **ACCURACY**: `{accuracy_score(y_test, y_pred):.3f}`"
        )

    st.divider()
    st.subheader("Classification report")

    y_pred = rf_model.predict(X)
    st.code(classification_report(y, y_pred))


def gradient_boosting():
    st.title("Gradient Boosting Classifier")

    st.subheader("Model parameters")

    learning_rate = st.number_input("Learning rate", value=0.1)
    n_estimators = st.number_input("Number of Estimators", value=100)
    criterion = st.selectbox("Criterion", ["friedman_mse", "squared_error"])

    st.divider()

    grad_model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    criterion=criterion,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                ),
            ),
        ]
    )

    st.code(grad_model)

    sss = StratifiedShuffleSplit(random_state=42)
    for i, (train_index, test_index) in enumerate(sss.split(X, y)):
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]

        grad_model.fit(X_train, y_train)
        y_pred = grad_model.predict(X_test)

        st.write(
            f"[`{i + 1:2}`] **LOSS**: `{mean_squared_error(y_test, y_pred):.3f}` **ACCURACY**: `{accuracy_score(y_test, y_pred):.3f}`"
        )

    st.divider()
    st.subheader("Classification report")

    y_pred = grad_model.predict(X)
    st.code(classification_report(y, y_pred))


pg = st.navigation(
    [
        st.Page(dataset_visualization, title="Dataset Visualization"),
        st.Page(logistic_regression, title="Logistic Regression"),
        st.Page(random_forest, title="Random Forest"),
        st.Page(gradient_boosting, title="Gradient Boosting"),
    ]
)
pg.run()
