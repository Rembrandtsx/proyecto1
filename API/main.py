from typing import Optional
import pandas as pd
from fastapi import FastAPI
from joblib import load
import DataModel
import PredictionModel
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("modelo.joblib")
    result = model.predict(df)
    print(model)
    return result.tolist()

cols = ["label", "study_and_condition"]

@app.post("/getrsquared")
def get_r2(dataList: DataModel.DataList):
    df = pd.DataFrame(columns=cols)
    for data in dataList.data:
        df = df.append(pd.DataFrame(np.array(list(data.dict().values())).reshape(1,-1), columns=cols), ignore_index=True)
    print(df)
    x = df.drop("label", axis=1)
    y = df["label"]
    model = load("modelo.joblib")
    result = model.score(x, y)
    return result

