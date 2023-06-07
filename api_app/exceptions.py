from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.status import HTTP_424_FAILED_DEPENDENCY


async def attribute_error_handler(
    _: Request,
    exc: AttributeError,
) -> JSONResponse:
    exc_message = str(exc)
    rsp_message = "Unknown error."

    if "factorizer" in exc_message:
        rsp_message = "Factorizer is not defined."
    elif "model" in exc_message:
        rsp_message = "Model is not defined."

    return JSONResponse(
        {"msg": rsp_message},
        status_code=HTTP_424_FAILED_DEPENDENCY,
    )
