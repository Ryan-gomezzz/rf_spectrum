"""
RF Spectrum Allocation - FastAPI Server
"""

from openenv.core.env_server import create_fastapi_app

from models import SpectrumAction, SpectrumObservation
from server.spectrum_environment import SpectrumEnvironment

env = SpectrumEnvironment()
app = create_fastapi_app(env, SpectrumAction, SpectrumObservation)
