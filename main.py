from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


titanicData = pd.read_csv('titanic_data.csv')

model = pickle.load(open('titanic_model_pickle.pkl', 'rb'))


class TitanicInfo(BaseModel):
    sexNo: int
    pClassNo: int
    ageNo: int
    nosbspNo: int
    noParchNo: int
    embarkLocNo: int


@app.post('/predict')
async def predict_survive(titanicInfo: TitanicInfo):

    sex = titanicInfo.sexNo
    pclass = titanicInfo.pClassNo
    age = titanicInfo.ageNo
    sbsp = titanicInfo.nosbspNo
    parch = titanicInfo.noParchNo
    embark = titanicInfo.embarkLocNo

    prediction = model.predict(pd.DataFrame([[pclass, sex, age, sbsp, parch, embark]],
                               columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    ))

    result = prediction[0]

    return { "resultOfPred" : str(result) }
