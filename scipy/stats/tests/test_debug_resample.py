import numpy as np
from scipy import stats
import inspect

# Create a simple KDE
np.random.seed(12345)
xn = np.random.randn(100)
gkde = stats.gaussian_kde(xn)

# Check the method signature
sig = inspect.signature(gkde.resample)
print(f"Method signature: resample{sig}")

# Try to call with seed
try:
    result = gkde.resample(10, seed=42)
    print(f"✅ SUCCESS: resample accepted seed parameter")
    print(f"Result shape: {result.shape}")
except TypeError as e:
    print(f"❌ FAILED: {e}")

# Check if it's using **kwargs
try:
    result = gkde.resample(10, some_random_param=123)
    print(f"⚠️  WARNING: resample accepted random parameter too!")
except TypeError as e:
    print(f"✅ GOOD: resample rejected random parameter: {e}")
