(df_model.head())
    lbe = LabelEncoder()
    df_copy = df_model.copy()
    df_model['Phases'] = lbe.fit_transform(df_model['Phases'])
    st.write("Phases Encoded")
    st.write(df_model.head())
    column_name = st.selectbox("Select a column", df_model.columns)
    column_data = df[co