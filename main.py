from typing import Optional
import pandas as pd
from fastapi import FastAPI
from joblib import load
import DataModel
import numpy as np
from models import preprocessor, encoder
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware


origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/logisticregression/predict")
def make_predictions(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    print(df)
    df.columns = dataModel.columns()
    X = df['study_and_condition']
    model = load("logisticregression.joblib")
    result = model.predict(X)
    print(model)
    return result.tolist()

@app.post("/naivebayes/predict")
def make_predictions(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    print(df)
    df.columns = dataModel.columns()
    X = df['study_and_condition']
    model = load("naivebayes.joblib")
    result = model.predict(X)
    print(model)
    return result.tolist()

@app.post("/svm/predict")
def make_predictions(dataModel: DataModel.DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    X = df['study_and_condition']
    model = load("svm.joblib")
    result = model.predict(X)
    print(model)
    return result.tolist()

cols = ["label", "study_and_condition"]

@app.post("/logisticregression/getrsquared")
def get_r2(dataList: DataModel.DataList):
    df = pd.DataFrame(columns=cols)
    for data in dataList.data:
        df = df.append(pd.DataFrame(np.array(list(data.dict().values())).reshape(1,-1), columns=cols), ignore_index=True)

    encoder(df)
    print(df)
    x = df['study_and_condition']
    y = df["label"].astype('int')
    model = load("logisticregression.joblib")
    result = model.score(x, y)
    return result

@app.post("/naivebayes/getrsquared")
def get_r2(dataList: DataModel.DataList):
    df = pd.DataFrame(columns=cols)
    for data in dataList.data:
        df = df.append(pd.DataFrame(np.array(list(data.dict().values())).reshape(1,-1), columns=cols), ignore_index=True)

    encoder(df)
    print(df)
    x = df['study_and_condition']
    y = df["label"].astype('int')
    model = load("naivebayes.joblib")
    result = model.score(x, y)
    return result

@app.post("/svm/getrsquared")
def get_r2(dataList: DataModel.DataList):
    df = pd.DataFrame(columns=cols)
    for data in dataList.data:
        df = df.append(pd.DataFrame(np.array(list(data.dict().values())).reshape(1,-1), columns=cols), ignore_index=True)

    encoder(df)
    print(df)
    x = df['study_and_condition']
    y = df["label"].astype('int')
    model = load("svm.joblib")
    result = model.score(x, y)
    return result

