# IT is USA_HOUSING APP MODEL
# before runing move it to C:\Users\USER\Desktop\capstone project\Houshing Regressoin   other wise error

import gradio as gr 
import pickle
import pandas as pd
import numpy as np

# Load models
model_names = [
    'LinearRegression' ,'LassoRegression' , 'RidgeRegression' ,'ElasticnetRegression' , 'SGDRegression' , 'PolynomialRegression',
    'DecisionTree', 'RanddomForest' , 'Kneighbors' , 'SVMRegression' ,'LGBM', 'XGBoost'
]

models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

# Load evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')

# Prediction function
def predict(model_name, income, house_age, rooms, bedrooms, population):
    input_data = {
        "Avg. Area Income": float(income),
        "Avg. Area House Age": float(house_age),
        "Avg. Area Number of Rooms": float(rooms),
        "Avg. Area Number of Bedrooms": float(bedrooms),
        "Area Population": float(population)
    }
    input_df = pd.DataFrame([input_data])
    model = models[model_name]
    prediction = model.predict(input_df)[0]
    return f"🏠 Predicted Price with {model_name}: ${prediction:,.2f}"

 
    
# Show evaluation results table
def show_results():
    return results_df



# Build Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🏡 Housing Price Prediction App")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(model_names, label="Select Model")
            income = gr.Number(label="Avg. Area Income")
            house_age = gr.Number(label="Avg. Area House Age")
            rooms = gr.Number(label="Avg. Area Number of Rooms")
            bedrooms = gr.Number(label="Avg. Area Number of Bedrooms")
            population = gr.Number(label="Area Population")
            predict_btn = gr.Button("Predict")
            output = gr.Textbox(label="Prediction Result")
        
        with gr.Column():
            gr.Markdown("### 📊 Model Evaluation Results")
            results_table = gr.DataFrame(value=results_df, label="Evaluation Metrics", interactive=False)
    
    predict_btn.click(
        fn=predict,
        inputs=[model_dropdown, income, house_age, rooms, bedrooms, population],
        outputs=output
    )

# Launch app
if __name__ == "__main__":
    demo.launch(share=False)






