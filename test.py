import os

import psutil

process = psutil.Process(os.getpid())
mem = process.memory_percent()