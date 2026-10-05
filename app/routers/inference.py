from fastapi import APIRouter, Depends

from app.controllers.inference import InferenceController, get_inference_controller
from app.schemas.inference import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])


@router.post("", response_model=InferenceResponse)
def infer(
    request: InferenceRequest,
    controller: InferenceController = Depends(get_inference_controller),
) -> InferenceResponse:
    return controller.infer(request)
