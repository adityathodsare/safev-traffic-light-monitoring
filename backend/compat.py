import torch
import sys

# Compatibility wrapper for older PyTorch versions
if not hasattr(torch.serialization, 'safe_globals'):
    # Define a dummy context manager for older PyTorch
    from contextlib import contextmanager
    
    @contextmanager
    def safe_globals(globals_list):
        """Dummy safe_globals for PyTorch < 2.2"""
        yield
    
    torch.serialization.safe_globals = safe_globals
    print("⚠️ Using compatibility mode for PyTorch < 2.2")