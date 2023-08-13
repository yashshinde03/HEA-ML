import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from pycaret.regression import setup, compare_models, pull, predict_model
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
showWarningOnDirectExecution = False
import pandas as pd
import numpy as np 
import pygwalker as pyg
from networkss import color_plot
data = pd.read_csv('data.csv')

# Load PyTorch model



with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("High Entropy Alloy")
    choice = st.selectbox("Navigation", ["Home","Dashboard","Regression","Classification","DeepThresholding"])
    st.info("This project application helps you to Predict the Properties and Analysis of Microstructure of High Entropy Alloys")
df = pd.read_csv("HEA.csv", encoding="latin-1")
drop_columns = ['Alloy ','HPR', 'Alloy ID', 'References', 'Al','Co','Cr','Fe','Ni','Cu','Mn','Ti','V',
                    'Nb','Mo','Zr','Hf','Ta','W','C','Mg','Zn','Si','Re','N','Sc','Li','Sn','Be',
                    'Unnamed: 51','Unnamed: 52','IM_Structure','Quenching','Annealing_Temp',
                    'Annealing_Time_(min)','Homogenization_Temp','Homogenization_Time',
                    'Sythesis_Route','dGmix','Hot-Cold_Working','Microstructure','Microstructure_',
                    'Multiphase','Tm','n.Para']

if choice == "Home":
    st.title("Welcome to HEA-App!")
    st.image("https://www.science.org/cms/10.1126/science.abn3103/asset/d965367e-895a-4dd1-a8c8-1924b2101bc5/assets/images/large/science.abn3103-f1.jpg")
    st.write(
        "#### Predict the Properties and Analysis of Microstructure of High Entropy Alloys!"
        )
    st.write(
        "This app mainly consists of 4 section - Home, Rergression, Classification and Deep Thresholding",
        "\n\n 1. Regression - The regression section involves predicting the properties of HEAs, including Enthalpy and Entropy. More details regarding each property can be found below.\n",
        "\n\n 2. Classification - The classification of phases in HEA alloys includes identifying and categorizing them into groups such as FCC, BCC, and Lm, among others.\n",
        "\n\n 3. Deep Thresholding - Deep thresholding is a technique used in image processing and computer vision applications to separate objects or features from the background by applying a threshold value or seperate the different phases in an Image"
        )
    st.markdown(
        "### Meet the HEA-App-Makers!\n\n|Name|GitHub Profile|\n|----|----|\n|Ankita Punekar|[ankitapunekar24](https://github.com/https://github.com/ankitapunekar24)|\n|Kiran Chandani|[123kiran17](https://github.com/123kiran17)|\n|Siddesh Thorat|[Mr-SidD](https://github.com/Mr-SidD)|\n|Yash Shinde|[yashshinde03](https://github.com/yashshinde03)|\n\n\n\n"
        )
    st.write("\n\n")
    st.markdown(
        "##### Get the detail implementation of Code!\n\n|Name|Colab Notebook|\n|----|----|\n|HEA Properties Prediction|[Colab Notebook](https://colab.research.google.com/drive/10buWv20Dw7jeGksNlptjZWbtF5W1fE5h?usp=sharing)|\n|Deep Thresholding|[Colab Notebook](https://colab.research.google.com/drive/1sLvZAaEX7_YrUnqAXPn9S6N1pSwUH4f7?usp=sharing)|\n\n\n\n"
        )
    st.write("\n\n\n\n\n")
    st.markdown(
        "\n\n##### Get the description of Properties here!\n\n|Name of Properties|Description|\n|----|----|\n|Num_of_Elem|Number of elements in the alloy system|\n|Density_calc|Calculated density of the alloy (in g/cm³).|\n|dHmix|Enthalpy of mixing for the alloy (in kJ/mol).|\n|dSmix|Entropy of mixing for the alloy (in J/mol/K).|\n|Atom.Size.Diff|Difference between atomic sizes of the two elements in the alloy|\n|Elect.Diff|Electronegativity difference between the two elements in the alloy.|\n|VEC|Valence electron concentration, a measure of metallic bonding strength.|\n|Phases|Crystal structure and phase of the alloy.|"
        )
    st.write("\n\n")
    df_model = pd.read_csv('NewHEA.csv')
    # Define the figure
    column_name = st.selectbox("Select a column", df_model.columns)
    column_data = df[column_name]
    st.write(column_data.head())
    fig = go.Figure()

    # Create the histogram trace
    histogram_trace = go.Histogram(x=df[column_name], nbinsx=20)

    # Add the trace to the figure
    fig.add_trace(histogram_trace)

    # Define the layout
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Frequency",
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="black",
    )

    # Draw the figure
    st.plotly_chart(fig)

if choice == "Dashboard":
    st.title("Dashboard")
    df = pd.read_csv('NewHEA.csv')
    widget = pyg.walk(df, env="Streamlit")

    # Render widget in Streamlit
    

if choice == "Regression":
    df = df.drop(drop_columns, axis=1)
    df.to_csv('NewHEA.csv', index=False)
    df_model = pd.read_csv('NewHEA.csv')
    st.write('Preprocess Dataset')
    st.write(df_model.head())
    lbe = LabelEncoder()
    df_copy = df_model.copy()
    df_model['Phases'] = lbe.fit_transform(df_model['Phases'])
    st.write("Phases Encoded")
    st.write(df_model.head())
    column_name = st.selectbox("Select a column", df_model.columns)
    column_data = df[column_name]
    st.write(column_data.head())
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=column_data, ax=ax)
    ax.set_title("Distribution of Phases")
    st.pyplot(fig)
    density_mean = df_model['Density_calc'].mean()
    df_model['Density_calc'] = df_model['Density_calc'].fillna(density_mean)
    density_mean = df_model['Num_of_Elem'].mode()
    df_model['Num_of_Elem'] = df_model['Num_of_Elem'].fillna(density_mean)
    last_row_index = df_model.index[-1]
    df_model = df_model.drop(index=last_row_index)
    df_model.info()
    df_model['Num_of_Elem'] = df_model['Num_of_Elem'].astype('int64')
    chosen_target = st.selectbox('Choose the Target Column', df_model.drop(['Phases'], axis = 1).columns)
    df_model = df_model.dropna(subset=[chosen_target])
    if st.button('Run Modelling'): 
        imputer = SimpleImputer(strategy='mean')
        df_model[chosen_target] = imputer.fit_transform(df_model[[chosen_target]])
        setup(df_model, target=chosen_target)
        setup_df = pull()
        st.write('Model Setup Info')
        st.dataframe(setup_df)
        st.write('Comparing the best model for training')
        best_model = compare_models()
        compare_df = pull()
        st.write('Best Model Comparison')
        st.dataframe(compare_df)
        trained_model = pull()
        predictions = predict_model(best_model, data=df_model)
        st.write('Predictions')
        st.dataframe(predictions.head())
        target_column = predictions.columns[-1]       
        actual_values = df_model[chosen_target]
        predicted_values = predictions[target_column]
        if actual_values.dtype == 'float' or predicted_values.dtype == 'float':
                predicted_values = predicted_values.astype(float)
        else:
            predicted_values = predicted_values.astype(int)
        
        mse = mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        st.write('Mean Squared Error:', mse)
        st.write('Mean Absolute Error:', mae)

if choice == "Classification":
    df = df.drop(drop_columns, axis=1)
    df.to_csv('NewHEA.csv', index=False)
    df_model = pd.read_csv('NewHEA.csv')
    st.write('Preprocess Dataset')
    st.write(df_model.head())
    lbe = LabelEncoder()
    df_copy = df_model.copy()
    df_model['Phases'] = lbe.fit_transform(df_model['Phases'])
    st.write("Phases Encoded")
    st.write(df_model.head())
    column_name = st.selectbox("Select a column", df_model.columns)
    column_data = df[column_name]
    st.write(column_data.head())
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=column_data, ax=ax)
    ax.set_title("Distribution of Phases")
    st.pyplot(fig)
    density_mean = df_model['Density_calc'].mean()
    df_model['Density_calc'] = df_model['Density_calc'].fillna(density_mean)
    density_mean = df_model['Num_of_Elem'].mode()
    df_model['Num_of_Elem'] = df_model['Num_of_Elem'].fillna(density_mean)
    last_row_index = df_model.index[-1]
    df_model = df_model.drop(index=last_row_index)
    df_model.info()
    from pycaret.classification import setup, compare_models, pull, predict_model
    # df_model['Num_of_Elem'] = df_model['Num_of_Elem'].astype('int64')
    chosen_target = st.selectbox('Choose the Target Column', df_model.drop(['Num_of_Elem','Density_calc','dHmix','dSmix','Atom.Size.Diff','Elect.Diff','VEC'], axis = 1).columns)
    df_model = df_model.dropna(subset=[chosen_target])
    if st.button('Run Modelling'): 
        imputer = SimpleImputer(strategy='mean')
        df_model[chosen_target] = imputer.fit_transform(df_model[[chosen_target]])
        setup(df_model, target=chosen_target)
        setup_df = pull()
        st.write('Model Setup Info')
        st.dataframe(setup_df)
        st.write('Comparing the best model for training')
        best_model = compare_models()
        compare_df = pull()
        st.write('Best Model Comparison')
        st.dataframe(compare_df)
        trained_model = pull()
        predictions = predict_model(best_model, data=df_model)
        st.write('Predictions')
        st.dataframe(predictions.head())
        target_column = predictions.columns[-2]       
        actual_values = df_model[chosen_target]
        predicted_values = predictions[target_column]
        accuracy = accuracy_score(actual_values, predicted_values)
        cm = confusion_matrix(actual_values, predicted_values)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot()
        report = classification_report(actual_values, predicted_values, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.write('Accuracy Score:', accuracy)

if choice == "DeepThresholding":
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png','tif'])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded image', use_column_width=True)
        filename = uploaded_file.name
    # Display the filename in Streamlit
        filename = filename[0:-4]
        index = data['Name'].tolist().index(filename)
        color_plot(index, [4,190,234])
        
    else:
        st.write('Please upload an image.')       
        
            