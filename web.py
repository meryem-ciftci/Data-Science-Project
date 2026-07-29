import streamlit as st
import pandas as p
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

mushroom_mappings ={
    'is-edible': {'e': 'Edible', 'p': 'Poisonous'},
    'cap-shape': {'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'},
    'cap-surface': {'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'},
    'cap-color': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'r': 'Green', 'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'bruises': {'t': 'Bruises (Var)', 'f': 'No Bruises'},
    'odor': {'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul', 'm': 'Musty', 'n': 'None', 'p': 'Pungent', 'sp': 'Spicy'},
    'gill-attachment': {'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'notched': 'Notched'},
    'gill-spacing': {'c': 'Close', 'w': 'Crowded', 'd': 'Distant'},
    'gill-size': {'b': 'Broad', 'n': 'Narrow'},
    'gill-color':{'k':'black','n':'brown','b':'buff','h':'chocolate','g':'gray', 'r':'green','o':'orange','p':'pink','u':'purple','e':'red', 'w':'white','y':'yellow'},
    'stalk-shape':{'e':'enlarging','t':'tapering'}, 
    'stalk-root':{'b':'bulbous','c':'club','u':'cup','e':'equal', 'z':'rhizomorphs','r':'rooted','?':'missing'},
    'stalk-surface-above-ring':{'f':'fibrous','y':'scaly','k':'silky','s':'smooth'}, 
    'stalk-surface-below-ring':{'f':'fibrous','y':'scaly','k':'silky','s':'smooth'},
    'stalk-color-above-ring':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange', 'p':'pink','e':'red','w':'white','y':'yellow'},
    'stalk-color-below-ring':{'n':'brown','b':'buff','c':'cinnamon','g':'gray','o':'orange', 'p':'pink','e':'red','w':'white','y':'yellow'},
    'veil-type':{'p':'partial','u':'universal'},
    'veil-color':{'n':'brown','o':'orange','w':'white','y':'yellow'},
    'ring-number':{'n':'none','o':'one','t':'two'},
    'ring-type':{'c':'cobwebby','e':'evanescent','f':'flaring','l':'large', 'n':'none','p':'pendant','s':'sheathing','z':'zone'},
    'spore-print-color':{'k':'black','n':'brown','b':'buff','h':'chocolate', 'r':'green','o':'orange','u':'purple','w':'white','y':'yellow'},
    'population':{'a':'abundant','c':'clustered','n':'numerous', 's':'scattered','v':'several','y':'solitary'},
    'habitat':{'g':'grasses','l':'leaves','m':'meadows','p':'paths', 'u':'urban','w':'waste','d':'woods'}
}

with st.sidebar:
    st.image("pngtree-red-mushroom-cartoon-vector-logo-png-image_16109385.png", width=100) # Mantar ikonu
    st.title("Analysis of Mushrooms and their Poison Rate")
    st.markdown("""
                
    **Group 28**
    
    **Group Members:**
    * **Damla Nur Kurt**
    * **Meryem Çiftçi**
    * **Yiğit Gürler**
    """)

def load_data():
    return p.read_csv("filtered.csv")

df = load_data()

prediction_df = p.read_csv('tüm_modeller_tahmin_analizi.csv')


st.title("Analysis of our Dataset")

m1,m2,m3 = st.columns(3)

with m1:
    st.metric(label="Total Number of Data", value=f"{len(df)} Num")
with m2:
    st.metric(label="Number of features", value=f"{len(df.columns)-1} Criteria")
with m3:
    poisonousCount=df[df['is-edible'].str.contains('poisonous|p', na=False)].shape[0]
    poisonousRatio=(poisonousCount/len(df))*100
    st.metric(label="Poison Rate", value=f"{poisonousRatio:.2f}")

st.markdown("---")

st.title("Analysis of a Random Mushroom")


for columns, mapping in mushroom_mappings.items():
    if columns in df.columns:
        df[columns]=df[columns].map(mapping).fillna(df[columns])


if st.button("Choose a random mushroom from dataset"):

    df = df.iloc[prediction_df.index]

    random_sample = df.sample(1)
    actual_class = random_sample['is-edible'].values[0]
    random_index = random_sample.index[0]

    st.write("All attributes of choosen mushroom:")

    st.dataframe(random_sample)

    row = prediction_df.iloc[random_index]
    st.write(f"Analysis of choosen mushroom with index **{random_index}:")

    col_left, col_right = st.columns(2)
    with col_left:
        if row['modelin_tahmini']==0:
            guess = "Poisonous"
        elif row['modelin_tahmini']==1:
            guess="Edible"
        st.info(f"**Model Guess:** {guess}")
    with col_right:
        if row['gercek_deger']==0:
            guess = "Poisonous"
        elif row['gercek_deger']==1:
            guess="Edible"
        st.warning(f"**True Value:** {guess}")

    if row['durum'] == 'Dogru':
        st.success(f"Result: Model guessed it perfectly!!")
    else:
        st.error(f"Result: Model made a mistake on this mushroom.")

    if "Poisonous" in actual_class or actual_class == 'p':
        st.error("POISONOUS")
    else:
        st.success("SAFE")


st.title("Bivariate Analysis")
st.write("Choose a category:")

all_features =[
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
    'stalk-shape', 'stalk-root', 'habitat', 'population', 'ring-number', 'ring-type',
    'spore-print-color','stalk-color-above-ring','stalk-color-below-ring','stalk-root',
    'stalk-shape','stalk-surface-above-ring','stalk-surface-below-ring','veil-color'
]

selected_analysis = st.selectbox("Choose one attribute:",all_features)

fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(data=df, x=selected_analysis, hue='is-edible', ax=ax, palette='Set1')

ax.set_title(f"{selected_analysis.capitalize()} Poison distribution of choosen attribute", fontsize=12, fontweight='bold')
ax.set_xlabel(selected_analysis, fontsize=10)
ax.set_ylabel("Number of Mushrooms", fontsize=10)
plt.xticks(rotation=0)
st.pyplot(fig)

st.info(f"As we can see the most significant factor is mushroom's odor.")

st.markdown("---")

prediction_df = p.read_csv('tüm_modeller_tahmin_analizi.csv')

st.title("Classification and Regression Report")

c1,c2,c3 = st.columns(3)

with c1:
    st.metric(label="Number of Simulation", value=f"{len(prediction_df)}")
with c2:
    dogru_sayisi = (prediction_df['durum'] == 'Dogru').sum()
    st.metric(label="Success Rate", value=f"{dogru_sayisi}")
with c3:
    basari_orani = (dogru_sayisi / len(prediction_df)) * 100
    st.metric(label="Accuracy Rate", value=f"{basari_orani:.2f}")

st.title("Univariate Analysis of Mushrooms")

all_features1 =[
    'cap-shape', 'cap-surface', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
    'stalk-shape', 'stalk-root', 'habitat', 'population', 'ring-number', 'ring-type',
    'spore-print-color','stalk-color-above-ring','stalk-color-below-ring','stalk-root',
    'stalk-shape','stalk-surface-above-ring','stalk-surface-below-ring','veil-color'
]

univairate_analysis = st.selectbox("Choose One", all_features1, key="univariate_image_select")

# 3. Resim yolunu güvenli şekilde oluşturup ekrana basıyoruz
if univairate_analysis:
    # os.path.join hem ters çizgi hatasını (SyntaxWarning) çözer hem de yolu güvenli birleştirir
    image_path = os.path.join("barCharts", f"{univairate_analysis}.png")
    
    # Resim klasörde var mı kontrol edip ekrana basıyoruz
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Distribution of {univairate_analysis.title()}", use_container_width=True)
    else:
        st.warning(f" {univairate_analysis}.png ERROR.")
    

