
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import re
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

model = None
symptom_columns = []
model_status = "Model not trained"
model_accuracy = None
demo_mode = False


def normalize(text):
    """Normalize symptom and column names."""
    return re.sub(
        r"[^a-z0-9]+", "_", str(text).lower()
    ).strip("_")


def prepare_features(df, columns):
    """Convert symptom features into numeric values."""
    X = df.reindex(columns=columns).copy()

    for col in columns:
        values = X[col]

        if pd.api.types.is_numeric_dtype(values):
            X[col] = pd.to_numeric(
                values, errors="coerce"
            ).fillna(0)
        else:
            X[col] = (
                values.astype("string")
                .str.strip()
                .str.lower()
                .map({
                    "1": 1,
                    "yes": 1,
                    "true": 1,
                    "present": 1,
                    "0": 0,
                    "no": 0,
                    "false": 0,
                    "absent": 0
                })
                .fillna(0)
            )

    return X.astype(float)


def train_model():
    """Train the final model and evaluate it on held-out data."""

    global model, symptom_columns
    global model_status, model_accuracy, demo_mode

    model = None
    symptom_columns = []
    model_accuracy = None
    demo_mode = False

    training_file = DATA_DIR / "Training.csv"
    testing_file = DATA_DIR / "Testing.csv"

    # Prefer the training dataset.
    if training_file.exists():
        train_file = training_file
        test_file = testing_file if testing_file.exists() else None
    elif testing_file.exists():
        # Fallback for demonstration only.
        train_file = testing_file
        test_file = None
        demo_mode = True
    else:
        model_status = "Dataset not found in the data folder"
        return

    try:
        # Load training dataset.
        train_df = pd.read_csv(train_file)
        train_df.columns = [
            normalize(col) for col in train_df.columns
        ]

        target_candidates = [
            "prognosis",
            "disease",
            "diagnosis",
            "target"
        ]

        target = next(
            (
                col for col in target_candidates
                if col in train_df.columns
            ),
            None
        )

        if target is None:
            model_status = (
                "Disease-label column not found. "
                "Expected prognosis, disease, diagnosis, or target."
            )
            return

        train_df = train_df.dropna(subset=[target])

        symptom_columns = [
            col for col in train_df.columns if col != target
        ]

        if (
            not symptom_columns
            or train_df[target].nunique() < 2
        ):
            symptom_columns = []
            model_status = (
                "Dataset needs symptom features and at least "
                "two disease classes."
            )
            return

        X = prepare_features(train_df, symptom_columns)
        y = train_df[target].astype(str)

        # Prepare a separate test dataset if available.
        X_test = None
        y_test = None

        if test_file is not None:
            test_df = pd.read_csv(test_file)
            test_df.columns = [
                normalize(col) for col in test_df.columns
            ]

            if target in test_df.columns:
                test_df = test_df.dropna(subset=[target])

                if not test_df.empty:
                    X_test = prepare_features(
                        test_df, symptom_columns
                    )
                    y_test = test_df[target].astype(str)

        # If no separate test data exists, use a stratified
        # holdout split when there are enough examples per class.
        X_fit = X
        y_fit = y

        if X_test is None:
            counts = y.value_counts()

            if len(y) >= 10 and counts.min() >= 2:
                X_fit, X_test, y_fit, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

        # Evaluate on data not used to fit the evaluation model.
        evaluation_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

        evaluation_model.fit(X_fit, y_fit)

        if X_test is not None and y_test is not None:
            # Evaluate only labels seen during training.
            known = y_test.isin(evaluation_model.classes_)

            if known.any():
                predictions = evaluation_model.predict(
                    X_test.loc[known]
                )

                model_accuracy = accuracy_score(
                    y_test.loc[known],
                    predictions
                )

        # Train the final prediction model on all training rows.
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X, y)

        model_status = f"Model trained using {train_file.name}"

        if demo_mode:
            model_status += " (demo dataset; not proper training)"
        elif model_accuracy is None:
            model_status += " (evaluation score unavailable)"
        elif test_file is not None:
            model_status += " (evaluated using separate test data)"
        else:
            model_status += " (evaluated using a holdout split)"

    except Exception as exc:
        model = None
        symptom_columns = []
        model_accuracy = None
        model_status = f"Model setup failed: {exc}"


def predict_disease(message):
    """Match dataset symptoms and generate an educational prediction."""

    if model is None:
        return None

    text = normalize(message)
    detected = []

    for symptom in symptom_columns:
        readable = symptom.replace("_", " ")

        if (
            re.search(
                r"(?<![a-z0-9])"
                + re.escape(symptom)
                + r"(?![a-z0-9])",
                text
            )
            or re.search(
                r"(?<![a-z0-9])"
                + re.escape(readable)
                + r"(?![a-z0-9])",
                message.lower()
            )
        ):
            detected.append(symptom)

    if not detected:
        return None

    # Simplified educational input:
    # unmentioned symptoms are treated as absent by this model.
    row = pd.DataFrame(
        [
            [int(col in detected) for col in symptom_columns]
        ],
        columns=symptom_columns
    )

    probabilities = model.predict_proba(row)[0]
    best_index = probabilities.argmax()

    return {
        "disease": str(model.classes_[best_index]),
        "confidence": round(
            float(probabilities[best_index]) * 100, 1
        ),
        "matched_symptoms": [
            symptom.replace("_", " ")
            for symptom in detected
        ]
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    symptoms = str(data.get("message", "")).strip().lower()

    if not symptoms:
        return jsonify({
            "response": "Please describe your symptoms."
        })

    if len(symptoms) > 1000:
        return jsonify({
            "response": "Please keep your message under 1000 characters."
        }), 400

    # Emergency screening before any model prediction.
    emergency_terms = [
        "chest pain",
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "shortness of breath",
        "unconscious"
    ]

    if any(term in symptoms for term in emergency_terms):
        return jsonify({
            "response": (
                "This may be a medical emergency. Contact your local "
                "emergency service or go to the nearest emergency "
                "department immediately. Do not rely on an AI prediction."
            )
        })

    # General information for common symptoms.
    advice = {
        "fever": (
            "Fever can have many causes. Stay hydrated and monitor "
            "your temperature. Seek medical advice if it is high, "
            "persistent, or accompanied by concerning symptoms."
        ),
        "cough": (
            "Cough can have several causes. Rest and stay hydrated. "
            "Seek medical care for worsening symptoms or breathing problems."
        ),
        "headache": (
            "Headaches have many possible causes. A sudden, severe, "
            "or unusual headache needs urgent medical assessment."
        ),
        "stomach pain": (
            "Stomach pain can have many causes. Severe or persistent "
            "pain, bleeding, or repeated vomiting needs medical care."
        ),
        "vomit": (
            "Vomiting can have several causes. Take small sips of fluid. "
            "Seek care for persistent vomiting or blood."
        ),
        "diarrhea": (
            "Diarrhea can cause dehydration. Drink fluids or oral "
            "rehydration solution and seek care for blood or severe symptoms."
        )
    }

    result = predict_disease(symptoms)

    if result:
        response = (
            "Machine Learning Screening Result (Educational Only)\n\n"
            f"Model class: {result['disease']}\n"
            f"Model score: {result['confidence']}%\n"
            "Matched symptoms: "
            f"{', '.join(result['matched_symptoms'])}\n\n"
            "Important: This score is not the probability that you have "
            "this disease and is not a confirmed diagnosis. The model "
            "uses a simplified representation of symptoms and may be "
            "wrong. Consult a qualified healthcare professional."
        )
    else:
        selected_advice = next(
            (
                text for keyword, text in advice.items()
                if keyword in symptoms
            ),
            None
        )

        response = selected_advice or (
            "Please describe your symptoms in more detail. You can mention "
            "fever, cough, headache, stomach pain, vomiting, or diarrhea. "
            "This chatbot provides general health information only."
        )

        response += (
            "\n\nML status: " + model_status +
            "\nIf the model cannot match your symptoms to dataset features, "
            "it cannot generate a symptom-based prediction."
        )

    return jsonify({"response": response})


@app.route("/model-status")
def get_model_status():
    return jsonify({
        "status": model_status,
        "accuracy": (
            round(model_accuracy * 100, 2)
            if model_accuracy is not None else None
        ),
        "demo_mode": demo_mode,
        "symptom_features": len(symptom_columns)
    })


# Train the model when the Flask application starts.
train_model()


if __name__ == "__main__":
    app.run(debug=True)
