REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .autoencoder_agent_v0 import AutoencoderV0
REGISTRY['autoencoderv0'] = AutoencoderV0

TRANFORMER_BASED_AGENTS = ['updet', 'autoencoderv0']