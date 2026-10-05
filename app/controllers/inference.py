from functools import lru_cache

from app.schemas.inference import InferenceRequest, InferenceResponse
from app.services.jev import JevService, get_jev_service


class InferenceController:
    def __init__(self, service: JevService) -> None:
        self._service = service

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        return self._service.infer(request)


@lru_cache
def get_inference_controller() -> InferenceController:
    return InferenceController(get_jev_service())
