from pydantic import BaseModel, Field


class PassengerFeatures(BaseModel):
    pclass: int = Field(
        ..., ge=1, le=3, description="Passenger class: 1=first, 2=second, 3=third"
    )
    sex: str = Field(..., pattern="^(male|female)$", description="'male' or 'female'")
    age: float | None = Field(
        None, ge=0, le=120, description="Age in years; null triggers median imputation"
    )
    sib_sp: int = Field(0, ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(0, ge=0, description="Number of parents/children aboard")
    fare: float = Field(0.0, ge=0, description="Ticket fare in GBP")
    embarked: str | None = Field(
        None,
        pattern="^[SCQ]$",
        description="Port: S=Southampton, C=Cherbourg, Q=Queenstown",
    )

    def to_model_row(self) -> dict:
        return {
            "Pclass": self.pclass,
            "Sex": self.sex,
            "Age": self.age,
            "SibSp": self.sib_sp,
            "Parch": self.parch,
            "Fare": self.fare,
            "Embarked": self.embarked,
        }


class PredictRequest(BaseModel):
    passengers: list[PassengerFeatures] = Field(
        ..., min_length=1, description="One or more passengers to score"
    )


class PredictionResult(BaseModel):
    survived: int = Field(..., description="0 = did not survive, 1 = survived")
    survival_probability: float = Field(
        ..., description="Model confidence score [0, 1]"
    )
    verdict: str = Field(..., description="Human-readable outcome")


class PredictResponse(BaseModel):
    model_type: str
    count: int
    predictions: list[PredictionResult]
