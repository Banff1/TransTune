import sys 
import signal
import time
sys.path.append("..") 

from optimizer_pobo_sa import PollingBayesianOptimization
from utils import RealEnv


if __name__ == '__main__':
    def _handle_signal(signum, _frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(f"[{ts}] Received signal {signame} ({signum}). Exiting...", flush=True)
        raise SystemExit(128 + int(signum))

    # IMPORTANT:
    # When running under `nohup` (or when SSH disconnects), a SIGHUP may be delivered.
    # If we register a handler for SIGHUP, we override nohup's default ignore behavior
    # and the tuner will exit unexpectedly. So we explicitly IGNORE SIGHUP here.
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        # SIGHUP may not be available/assignable on all platforms.
        pass

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            # Some signals may not be available/assignable on all platforms.
            pass

    # ============================================
    # 配置参数：在这里指定要测试的数据集
    # ============================================
    DATASET = "dbpedia-openai-1M-1536-angular"
    
    # prepare the environment
    env = RealEnv(dataset=DATASET)
    model = PollingBayesianOptimization(env, seed=1)
    
    # initial sampling
    model.init_sample()

    # iterative auto-tuning
    for i in range(200):
        model.step()
