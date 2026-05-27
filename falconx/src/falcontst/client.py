from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import requests


DEFAULT_PREDICT_URL = (
    "https://falconstudio-pre.antglobal.com"
    "/falconstudio/api/v1/openapi/predict"
)
DEFAULT_BATCH_PREDICT_URL = (
    "https://falconstudio-pre.antglobal.com"
    "/falconstudio/api/v1/openapi/batch-predict"
)


class FalconAPIError(Exception):
    """Raised when Falcon Studio API returns a non-success code."""


class FalconClient:
    """Client for Falcon Studio prediction API."""

    def __init__(
        self,
        endpoint: str = DEFAULT_PREDICT_URL,
        batch_endpoint: str = DEFAULT_BATCH_PREDICT_URL,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
        verify: Optional[Union[bool, str]] = None,
    ) -> None:
        self.endpoint = endpoint
        self.batch_endpoint = batch_endpoint
        self.timeout = timeout
        self.session = session or requests.Session()
        self.verify = verify

    def quantile_predict(
        self,
        context: np.ndarray,
        prediction_length: int,
        model_name: Optional[str] = None,
        group_ids: Optional[np.ndarray] = None,
        input_mask: Optional[np.ndarray] = None,
        is_multivariate: bool = False,
    ) -> Any:
        """Quantile forecast inference."""
        payload = self._build_predict_payload(
            context=context,
            prediction_length=prediction_length,
            model_name=model_name,
            group_ids=group_ids,
            input_mask=input_mask,
            is_multivariate=is_multivariate,
        )

        return self._post(self.endpoint, payload)

    def batch_predict(self, objects: Sequence[Mapping[str, Any]]) -> Any:
        """Batch quantile forecast inference."""
        if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
            raise TypeError(
                f"Expected objects to be a sequence, got {type(objects).__name__}"
            )

        payload = []
        for item in objects:
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Expected batch item to be a mapping, got {type(item).__name__}"
                )

            payload.append(
                self._build_predict_payload(
                    context=item["context"],
                    prediction_length=item["prediction_length"],
                    model_name=item.get("model_name"),
                    group_ids=item.get("group_ids"),
                    input_mask=item.get("input_mask"),
                    is_multivariate=item.get("is_multivariate", False),
                )
            )

        return self._post(self.batch_endpoint, payload)

    def _build_predict_payload(
        self,
        context: np.ndarray,
        prediction_length: int,
        model_name: Optional[str] = None,
        group_ids: Optional[np.ndarray] = None,
        input_mask: Optional[np.ndarray] = None,
        is_multivariate: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(prediction_length, int):
            raise TypeError(
                f"Expected prediction_length to be int, got {type(prediction_length).__name__}"
            )
        if prediction_length <= 0:
            raise ValueError("prediction_length must be greater than 0")

        payload = {
            "context": self._array_to_list(context),
            "prediction_length": prediction_length,
            "is_multivariate": is_multivariate,
        }

        if model_name is not None:
            payload["model_name"] = model_name
        if group_ids is not None:
            payload["group_ids"] = self._array_to_list(group_ids)
        if input_mask is not None:
            payload["input_mask"] = self._array_to_list(input_mask)

        return payload

    def _post(self, endpoint: str, payload: Any) -> dict[str, Any]:
        request_kwargs = {
            "json": payload,
            "timeout": self.timeout,
        }
        if self.verify is not None:
            request_kwargs["verify"] = self.verify

        response = self.session.post(endpoint, **request_kwargs)
        response.raise_for_status()

        return self._format_response(response.json())

    @staticmethod
    def _array_to_list(array: np.ndarray) -> list[Any]:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(array).__name__}")
        return FalconClient._replace_nan(array.tolist())

    @staticmethod
    def _replace_nan(value: Any) -> Any:
        if isinstance(value, list):
            return [FalconClient._replace_nan(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    @staticmethod
    def _format_response(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise FalconAPIError(f"Unexpected response format: {data}")

        code = data.get("code")
        message = data.get("message", "")
        if code != 200:
            raise FalconAPIError(f"Falcon API error: code={code}, message={message}")

        return {"prob_prediction": data.get("data")}
