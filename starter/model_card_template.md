# Model Card

## Model Details
This model predicts whether a person's income exceeds $50K per year using census data.

The model was trained using the Adult Census Income dataset.

Model type: Random Forest Classifier

Frameworks used:
- Scikit-learn
- Pandas
- NumPy

## Intended Use
This model is intended for educational purposes and demonstration of ML deployment using FastAPI.

The model predicts income categories:
- <=50K
- >50K

## Training Data
The dataset contains demographic and employment-related information such as:
- age
- education
- occupation
- hours-per-week
- marital-status

Target variable:
- salary

## Evaluation Data
20% of the dataset was used as test data.

## Metrics
Precision: 0.7391

Recall: 0.6384

Fbeta Score: 0.6851

## Ethical Considerations
The dataset may contain bias related to gender, race, and occupation. Predictions should not be used for real-world hiring or financial decisions.

## Caveats and Recommendations
This model is trained on limited census data and may not generalize well to real-world scenarios.