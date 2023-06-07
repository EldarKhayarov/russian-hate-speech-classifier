from pydantic import BaseModel


class PredictUploadSchema(BaseModel):
    comment: str


class PredictResponseSchema(BaseModel):
    toxic: int
