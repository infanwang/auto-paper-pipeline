#!/usr/bin/env python3
"""测试运行器"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """运行所有测试"""
    test_dir = Path(__file__).parent
    
    print("="*60)
    print("🧪 Auto Paper Pipeline - 测试套件")
    print("="*60)
    
    # 运行pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
