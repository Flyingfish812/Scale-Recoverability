"""pytest 公共配置: 将仓库根加入 sys.path, 使 `import luna` 可用。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
