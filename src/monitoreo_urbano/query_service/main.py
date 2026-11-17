from __future__ import annotations

import uvicorn

from monitoreo_urbano.common.config import settings


def main() -> None:
    uvicorn.run(
        "monitoreo_urbano.query_service.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
