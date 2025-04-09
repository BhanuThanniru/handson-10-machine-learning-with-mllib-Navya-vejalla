from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Helper to save formatted rows to a file
def save_formatted_output(rows, path, header1, header2):
    with open(path, "w") as f:
        f.write("+--------------------+-----------+\n")
        f.write(f"|{header1:<20}|{header2:<11}|\n")
        f.write("+--------------------+-----------+\n")
        for row in rows:
            f.write(f"|{str(row[0]):<20}|{str(row[1]):<11}|\n")
        f.write("+--------------------+-----------+\n")

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.fillna({'TotalCharges': 0})

    categorical_cols = ["gender", "PhoneService", "InternetService", "Churn"]
    indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols[:-1]]

    for indexer in indexers:
        df = indexer.fit(df).transform(df)
    for encoder in encoders:
        df = encoder.fit(df).transform(df)

    feature_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + [col + "Vec" for col in categorical_cols[:-1]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    df = df.withColumnRenamed("ChurnIndex", "label")

    rows = df.select("features", "label").take(5)
    save_formatted_output(rows, "outputs/task1_features.txt", "features", "label")

    return df.select("features", "label")

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)

    with open("outputs/task2_logistic_auc.txt", "w") as f:
        f.write(f"Logistic Regression Model Accuracy (AUC): {auc:.2f}\n")

    rows = predictions.select("features", "prediction").take(5)
    save_formatted_output(rows, "outputs/task2_predictions.txt", "features", "prediction")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", labelCol="label", outputCol="selectedFeatures")
    result = selector.fit(df).transform(df)
    rows = result.select("selectedFeatures", "label").take(5)
    save_formatted_output(rows, "outputs/task3_selected_features.txt", "selectedFeatures", "label")

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # Split data
    # Define models
    # Define hyperparameter grids
    # Perform cross-validation for each model
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label")

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "GBT": GBTClassifier(featuresCol="features", labelCol="label")
    }

    param_grids = {
        "LogisticRegression": ParamGridBuilder().addGrid(models["LogisticRegression"].regParam, [0.01, 0.1]).addGrid(models["LogisticRegression"].maxIter, [10, 20]).build(),
        "DecisionTree": ParamGridBuilder().addGrid(models["DecisionTree"].maxDepth, [5, 10]).build(),
        "RandomForest": ParamGridBuilder().addGrid(models["RandomForest"].maxDepth, [10, 15]).addGrid(models["RandomForest"].numTrees, [20, 50]).build(),
        "GBT": ParamGridBuilder().addGrid(models["GBT"].maxDepth, [5, 10]).addGrid(models["GBT"].maxIter, [10, 20]).build()
    }

    os.makedirs("outputs/task4", exist_ok=True)
    results_file = open("outputs/task4/model_comparison_results.txt", "w")

    for name, model in models.items():
        results_file.write(f"Tuning {name}...\n")
        grid = param_grids[name]
        cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        predictions = best_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        results_file.write(f"{name} Best Model Accuracy (AUC): {auc:.2f}\n")
        tuned_params = [param.name for param in grid[0].keys()]  # get tuned param names
        best_params = best_model.extractParamMap()
        param_str = ", ".join([
            f"{param.name}={value}" 
            for param, value in best_params.items() 
            if param.name in tuned_params and param.parent == best_model.uid
        ])
        results_file.write(f"Best Params for {name}: {param_str}\n\n")

        # Save sample predictions
        with open(f"output/task4/{name}_predictions_sample.txt", "w") as pred_file:
            pred_file.write("+--------------------+-----------+\n")
            pred_file.write("|features            |prediction |\n")
            pred_file.write("+--------------------+-----------+\n")
            for row in predictions.select("features", "prediction").take(5):
                pred_file.write(f"|{str(row.features):<20}|{row.prediction:<11}|\n")
            pred_file.write("+--------------------+-----------+\n")

# Execute tasks
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()