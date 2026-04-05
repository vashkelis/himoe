# Patch inspect.getsource to handle PyTorch builtins without source
# This prevents OSError when mmengine/torch tries to get source for C++ extensions
import inspect
_original_getsource = inspect.getsource
def _patched_getsource(obj):
    try:
        return _original_getsource(obj)
    except (OSError, TypeError):
        return "def _torch_builtin_(): pass"
inspect.getsource = _patched_getsource

from .himoe_ffn import HiMoEFFN
from .himoe_losses import HiMoEAuxLossHook
from .himoe_hooks import SaveExpertUsageHook
