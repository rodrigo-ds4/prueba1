#!/usr/bin/env python3
"""
Test to verify that gaussian_kde.resample does NOT support seed parameter
in the old version, and then we'll fix it.
"""

import numpy as np
from scipy import stats
from numpy.testing import assert_allclose
import pytest

def test_resample_no_seed_support():
    """Test that current resample method does NOT support seed parameter."""
    print("Testing that resample() does NOT accept seed parameter...")
    
    np.random.seed(12345)
    n_basesample = 100
    xn = np.random.randn(n_basesample)
    
    gkde = stats.gaussian_kde(xn)
    n_sample = 50
    
    # This should fail with TypeError because resample doesn't accept seed
    try:
        samp1 = gkde.resample(n_sample, seed=42)
        print("❌ ERROR: resample() accepted seed parameter when it shouldn't!")
        return False
    except TypeError as e:
        if "unexpected keyword argument" in str(e) and "seed" in str(e):
            print(f"✅ EXPECTED FAILURE: {e}")
            return True
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False

if __name__ == "__main__":
    print("🧪 Testing gaussian_kde.resample seed support...")
    print("=" * 50)
    
    test1 = test_resample_no_seed_support()
    
    print("=" * 50)
    if test1:
        print("✅ Confirmed: resample() does NOT support seed parameter")
        print("📝 Next step: Implement seed parameter support")
    else:
        print("❌ Unexpected: resample() seems to support seed already")
