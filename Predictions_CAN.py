import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

world_cup_csv = pd.read_csv('WorldCupMatches.csv')

world_cup = world_cup_csv[["Home Team Name","Home Team Goals", "Away Team Goals","Away Team Name"]]

world_cup = world_cup.dropna(how='all',axis=0) 

world_cup = world_cup.rename(columns={"Home Team Name": "home_team", "Home Team Goals": "home_score","Away Team Name": "away_team", "Away Team Goals": "away_score"})

world_cup["home_score"] = world_cup["home_score"].astype(int)

world_cup["away_score"] = world_cup["away_score"].astype(int)

can_hist_csv = pd.read_csv('African-Nations-results.csv')

can_hist_csv = can_hist_csv.drop(["date", "tournament"],axis=1)

can_hist_2025 = pd.read_csv('results.csv')

full = pd.concat([world_cup, can_hist_csv, can_hist_2025], ignore_index=True)

full["home_score"] = full["home_score"].astype(int)
full["away_score"] = full["away_score"].astype(int)

x = full.drop(["home_score","away_score"],axis=1)
y = full[["home_score","away_score"]]

categorical_features = ["home_team","away_team"]
one_hot = OneHotEncoder(handle_unknown="ignore")
preprocessor = ColumnTransformer([(
    "one_hot", 
    one_hot,
    categorical_features)],remainder="passthrough")

model = RandomForestRegressor(n_estimators=200,
    random_state=42,
    n_jobs=-1)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

#Split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2)

#Fit
pipe.fit(x_train, y_train)


#pipe.score(x_test,y_test)
#Score
#print(pipe.score(x_test,y_test))


def predire_match(home, away):
    """
    ligne_raw : dict ou pd.Series avec au moins les colonnes de X.
    Retourne un pd.Series (home_score, away_score).
    """
    import pandas as pd
    
    ligne_raw = {
        "home_team": home,
        "away_team": away
    } 
    
    if isinstance(ligne_raw, dict):
        ligne_df = pd.DataFrame([ligne_raw])
    else:
        ligne_df = ligne_raw.to_frame().T

    # S’assure de l’ordre des colonnes
    ligne_df = ligne_df[x.columns]

    y_pred = pipe.predict(ligne_df)  # shape (1, 2)
    return pd.Series(y_pred[0], index=y.columns)


matches = pd.read_csv('matches.csv')

def appliquer_predictions(matches):
    # apply ligne par ligne, une seule prédiction par match
    preds = matches.apply(
        lambda row: predire_match(row["home_team"], row["away_team"]),
        axis=1
    )
    # preds est un DataFrame avec colonnes home_score, away_score (index = y.columns)
    matches[["home_score", "away_score"]] = preds.round().astype(int)
    return matches

matches = appliquer_predictions(matches)
print(matches)
