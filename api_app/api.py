import pickle

from fastapi import UploadFile, Request
from fastapi.routing import APIRouter

from api_app.schemas import PredictUploadSchema, PredictResponseSchema
from api_app.services import make_prediction


router = APIRouter()


@router.post("/upload-factorizer")
async def upload_factorizer(request: Request, factorizer_upload: UploadFile):
    factorizer = pickle.load(factorizer_upload.file)
    request.app.state.factorizer = factorizer


@router.post("/upload-model")
async def upload_factorizer(request: Request, model_upload: UploadFile):
    model = pickle.load(model_upload.file)
    request.app.state.model = model


@router.post("/predict", response_model=PredictResponseSchema)
async def predict(request: Request, raw_features: PredictUploadSchema):
    factorizer = getattr(request.app.state, "factorizer")
    model = getattr(request.app.state, "model")
    return {"toxic": await make_prediction(factorizer, model, raw_features.comment)}
