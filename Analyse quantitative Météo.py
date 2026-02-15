
import numpy as np
import yfinance as yf
import pandas as pd
import meteostat as ms
from datetime import date

tickers = ["^GSPC", "^VIX", "NG=F", "XLU", "CORN"]
noms = tickers
def donnees_finance(tickers,noms):
    D = dict(zip(tickers, noms))
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
    df = df.rename(columns=D)
    # "^GSPC": "SP500",
    # "^VIX": "VIX",
    # "^NG=F": "NG=F",   
    # "XLU": "ELEC",
    # "CORN": "CORN"
    # })
    return df

df = donnees_finance(tickers,noms)
Datafinance = df
#Returns des actifs 
returns = Datafinance.pct_change()


START = date(2006, 1, 1)
END   = date(2026, 1, 1)

# Central Park station ID NOAA
# station_id = "72503"

#Champs de Mais champain Illinois ID KCMI
station_id = "72530"

def donnee_meteo(START,END,station_id):
    
#Données météo

    ts = ms.daily(station_id, START, END)
    df = ts.fetch()
    return df

#Jours d'ouverture des marchés
openday = [d.strftime("%Y-%m-%d") for d in df.index]
df = donnee_meteo(START, END, station_id)
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

def affiche_impact_meteo(nom):
    T.loc[nom].plot(kind="bar")
    plt.ylabel("Rendement")
    plt.xlabel("Variables météo")
    plt.title("Impact météo sur " + nom)
    plt.show()

affiche_impact_meteo("CORN")


#Manifestation d'évenements binaires, Neige = 1 pas de Neige = 0
#Objectif corelation avec les returns des actifs

def impact_evenement(returns ,liste_evenement):
    r = len(returns.index)
    e = len(liste_evenement)
    Z = np.zeros((r,e))
    for d in range(r):
        for i in range(e):
            if returns.index[d] in liste_evenement[i].index:
                Z[d,i] = 1
    df = pd.DataFrame(Z,returns.index,N)
    D = df.join(returns)
    return D

D = impact_evenement(returns, L)
print(D)

import statsmodels.api as sm


X = D[["rain", "snow", "wind", "cloud", "hum", "NG=F", "XLU"]]
# X = D[["snow"]]
y = D["CORN"]

#Régression linéaire
X = sm.add_constant(X)
model = sm.OLS(y, X, missing="drop").fit()
print(model.summary())


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Modèle de prévision d'un actif en fonction de la météo
def prevision_up(nom,returns, liste_evenement):
    

    Df = impact_evenement(returns, liste_evenement)
    N = list(Df.columns)
    r = len(returns.index)
    l = len(Df.columns)
    e = len(liste_evenement)
    #Modèle binaire d'évenement météo et de retour up = 1 down = 0, pour prevoir le futur
    Z = np.zeros((r,l))
    returns = returns.shift(-1)
    for d in range(r):
        for i in range(e):
            if returns.index[d] in liste_evenement[i].index:
                Z[d,i] = 1
        for j in range(e,l):
            if returns[N[j]].loc[returns.index[d]]>0:
                Z[d,j] = 1
    df = pd.DataFrame(Z,returns.index,N)
    data = pd.concat([df[[c for c in N if c != nom]],df[nom]],axis = 1).dropna()

    X = data.drop(columns = [nom])
    y = data[nom]
    
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



prevision_up("CORN", returns, L)

