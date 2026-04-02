"""
RF Spectrum Allocation - FastAPI Server
"""

import os
import sys

# Ensure project root is on the path so models/scenarios are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app

from models import SpectrumAction, SpectrumObservation
from server.spectrum_environment import SpectrumEnvironment

app = create_fastapi_app(SpectrumEnvironment, SpectrumAction, SpectrumObservation)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
