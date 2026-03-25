from titans_pytorch.neural_memory import (
    NeuralMemory,
    NeuralMemState,
    mem_state_detach
)

from titans_pytorch.memory_models import (
    MemoryMLP,
    MemoryAttention,
    FactorizedMemoryMLP,
    MemorySwiGluMLP,
    GatedResidualMemoryMLP
)

from titans_pytorch.mac_transformer import (
    MemoryAsContextTransformer
)

from titans_pytorch.memory_models_memristor import (
    MemristorConfig,
    MemristorMVM,
    MemoryMLP_Memristor,
    MemoryAttention_Memristor,
    GatedResidualMemoryMLP_Memristor,
    FactorizedMemoryMLP_Memristor,
    MemorySwiGluMLP_Memristor,
)
