This project leverages Machine Learning to predict Click-Through Rates (CTR) for digital marketing campaigns and provides actionable recommendations for budget reallocation. The system identifies high-performing campaigns to increase investment, flags underperforming campaigns to reduce spend, and provides AI-generated explanations for each recommendation.

It’s built for marketing teams, data analysts, and campaign managers who want to optimize ad spend efficiently.

Key Features

CTR Prediction: Predict future performance of campaigns using a trained Random Forest model.

ROI-Based Recommendations: Suggest actions such as Increase Budget, Monitor, or Pause/Reduce Budget based on campaign ROI.

Automated Budget Reallocation: Reallocate budget from low-performing campaigns to high-performing campaigns dynamically.

AI Explanations: Provide interpretable AI reasoning for every recommendation.

Interactive Streamlit Dashboard: Upload your campaign CSV and instantly visualize insights.

Tech Stack

Python 3.x

Pandas, NumPy – Data processing

Scikit-learn – Machine learning (Random Forest Regressor)

Streamlit – Interactive web dashboard

Joblib – Model and feature persistence

Optional: LLM-based explainability module for AI-generated explanations

Usage

Clone the repository:

git clone https://github.com/yourusername/ai-marketing-decision-engine.git
cd ai-marketing-decision-engine


Install dependencies:

pip install -r requirements.txt


Run the Streamlit dashboard:

streamlit run app.py


Upload your data.csv and view:

Predicted CTR

Budget recommendations

AI explanations for top campaigns
