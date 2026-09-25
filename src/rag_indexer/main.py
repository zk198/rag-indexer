from .config import Settings
from .indexer import run_forever
if __name__ == "__main__":
    run_forever(Settings.from_env())
