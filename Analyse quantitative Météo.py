

import numpy as np
import yfinance as yf
import pandas as pd
import meteostat as ms
from datetime import date

tickers = ["^GSPC", "^VIX", "NG=F", "XLU", "CORN"]

# Télécharger les données
df = yf.download(
    tickers,
    start="2006-01-01",
    end="2026-01-01",
    interval="1d"
)

# Garder UNIQUEMENT les prix de clôture
df = df.dropna(how="any")
df = df["Close"]

# Renommer les colonnes
df = df.rename(columns={
    "^GSPC": "SP500",
    "^VIX": "VIX",
    "^NG=F": "NG=F",   
    "XLU": "ELEC",
    "CORN": "CORN"
})

#Jours d'ouverture des marchés
openday = [d.strftime("%Y-%m-%d") for d in df.index]

Datafinance = df

#Données météo

START = date(2006, 1, 1)
END   = date(2026, 1, 1)

# Central Park station ID NOAA
# station_id = "72503"

#Champs de Mais champain Illinois ID KCMI
station_id = "72530"

ts = ms.daily(station_id, START, END)
df = ts.fetch()

# print(df.head())
# print(df.columns)

df = df.loc[openday]

#print(df[df['tsun']!= '<NA>'])


#Définition des catégories d'évenements météo
snow = df[df['snwd']>0]
rain = df[df['prcp']>20]
wind = df[df['wspd']>30]
cloud = df[df['cldc']>5]
tb=df[(df['tmin'] > 20) & (df['cldc'] < 4)] #temps beau, pas de nuage T>20°C
hum = df[df['rhum']>75] #Humidité dans l'air
L = [snow,rain,wind,cloud,tb,hum]
N = ['snow','rain','wind','cloud','tb','hum','norm']

#Création d'un évenement jour normal
Date = []
for dates in df.index:
    #df à toute les dates, on veut créer jour normal comme étant un jour sans évenements
    S = 0
    for l in L:
        if dates not in l.index:
            S = S+1
    if S == 6:
        Date.append(dates)  

Date = [d.strftime("%Y-%m-%d") for d in Date] #format date
norm = df.loc[Date]
L.append(norm)

#Liste des dates pour chaques évenements
DA = []
for l in L:
     DA.append([d.strftime("%Y-%m-%d") for d in l.index])
 

#Returns des actifs 
returns = Datafinance.pct_change()


#Rendements moyens dans une liste de dataframes
K = []
for d in range(len(DA)):
    K.append((returns.loc[DA[d]]).mean())

T = pd.concat(K,axis = 1)   
T = T.rename(columns={
    0: "snow",
    1: "rain",
    2: "wind",
    3: "cloud",
    4: "tb",
    5: "hum",
    6: "norm"
})


from pylab import plt,mpl

T.loc["CORN"].plot(kind="bar")
plt.ylabel("Rendement")
plt.xlabel("Variables météo")
plt.title("Impact météo sur CORN")
plt.show()


#Manifestation d'évenements binaires, Neige = 1 pas de Neige = 0
#Objectif corelation avec les returns des actifs
Z = np.zeros((len(returns.index),7))
for d in range(len(returns.index)):
    for i in range(7):
        if returns.index[d] in L[i].index:
            Z[d,i] = 1
            
df = pd.DataFrame(Z,returns.index,N)

D = df.join(returns)


import statsmodels.api as sm

X = D[["rain", "snow", "wind", "cloud", "hum", "NG=F", "ELEC"]]
# X = D[["snow"]]
y = D["CORN"]

#Régression linéaire
X = sm.add_constant(X)
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Maïs
N = ['snow','rain','wind','cloud','tb','hum','norm',
    'up_SP500','up_VIX','up_NG','up_XLU','up_CORN']

#Modèle binaire d'évenement météo et de retour up = 1 down = 0, pour prevoir le futur
Z = np.zeros((len(returns.index),12))
returns = returns.shift(-1)

for d in range(len(returns.index)):
    for i in range(7):
        if returns.index[d] in L[i].index:
            Z[d,i] = 1
    for j in range(7,12):
        if returns["CORN"].loc[returns.index[d]]>0:
            Z[d,j] = 1
df = pd.DataFrame(Z,returns.index,N)
data = pd.concat([df[N[:1]],df["up_CORN"]],axis = 1).dropna()


X = data[N[:1]]
y = data["up_CORN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)


rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))






