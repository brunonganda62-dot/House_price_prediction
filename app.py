from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "house_price_model.sav"

RANGES = {
    "Area_m2":             {"min": 26.0, "max": 426.4, "default": 105.7},
    "Bedrooms":            {"min": 1,    "max": 6,     "default": 3},
    "Bathrooms":           {"min": 1,    "max": 5,     "default": 3},
    "House_Age_Years":     {"min": 0.4,  "max": 34.3,  "default": 6.9},
    "Distance_to_City_km": {"min": 0.07, "max": 27.4,  "default": 3.3},
    "Parking_Spaces":      {"min": 0,    "max": 3,     "default": 1},
}
NEIGHBORHOODS = ["Gasabo", "Huye", "Kicukiro", "Kigali City", "Musanze", "Nyarugenge"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write(
    "Enter the details of a house in Rwanda and get an instant estimate of its market "
    "price in **million RWF**, using a multiple linear regression model trained on past sales."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file `house_price_model.sav` was not found next to app.py.")
    st.stop()
except Exception as e:
    st.error(f"The model could not be loaded: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (m²)", min_value=10.0, max_value=1000.0,
                           value=float(RANGES["Area_m2"]["default"]), step=1.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15,
                               value=int(RANGES["Bedrooms"]["default"]), step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=15,
                                value=int(RANGES["Bathrooms"]["default"]), step=1)
    parking = st.number_input("Parking spaces", min_value=0, max_value=10,
                              value=int(RANGES["Parking_Spaces"]["default"]), step=1)
with col2:
    age = st.number_input("House age (years)", min_value=0.0, max_value=100.0,
                          value=float(RANGES["House_Age_Years"]["default"]), step=0.5)
    distance = st.number_input("Distance to city centre (km)", min_value=0.0, max_value=100.0,
                               value=float(RANGES["Distance_to_City_km"]["default"]), step=0.5)
    neighborhood = st.selectbox("Neighbourhood", NEIGHBORHOODS)

if st.button("Predict", type="primary"):
    new_house = pd.DataFrame([{
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking,
        "Neighborhood": neighborhood,
    }])

    inputs = {
        "Area_m2": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "House_Age_Years": age,
        "Distance_to_City_km": distance,
        "Parking_Spaces": parking,
    }
    outside = [
        f"{name} = {value} (training range {RANGES[name]['min']} to {RANGES[name]['max']})"
        for name, value in inputs.items()
        if not RANGES[name]["min"] <= value <= RANGES[name]["max"]
    ]
    if outside:
        st.warning(
            "Some values are outside the range the model was trained on, so the estimate "
            "may be unreliable:\n\n- " + "\n- ".join(outside)
        )
    if bedrooms > 0 and bathrooms == 0:
        st.info("A house with bedrooms but no bathroom is unusual; please check your input.")

    try:
        price = float(model.predict(new_house)[0])
    except Exception as e:
        st.error(f"The prediction failed: {e}")
        st.stop()

    if price <= 0:
        st.error("The model returned a non-positive price for these inputs. "
                 "Please enter values closer to typical houses.")
    else:
        st.success(f"Estimated price: **{price:,.1f} million RWF**")
        st.caption(f"≈ {price * 1_000_000:,.0f} RWF")

with st.expander("About this model"):
    st.write(
        "Multiple linear regression (scikit-learn pipeline with median imputation and "
        "one-hot encoding). Test R² ≈ 0.78 and RMSE ≈ 24 million RWF, so treat the result "
        "as an estimate, not a valuation."
    )