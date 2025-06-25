from fastapi import FastAPI
import psutil
import time
import logging
from datetime import datetime
from typing import List, Dict
import os

app = FastAPI()


class SystemMonitor:
    def __init__(self):
        self.log_file = "system_monitor.log"
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_system_stats(self) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
        }

    def log_stats(self):
        stats = self.get_system_stats()
        logging.info(
            f"CPU: {stats['cpu_percent']}% | "
            f"Memory: {stats['memory_usage']}% | "
            f"Disk: {stats['disk_usage']}%"
        )
        return stats


monitor = SystemMonitor()


@app.get("/stats")
async def get_current_stats():
    return monitor.get_system_stats()


@app.get("/logs")
async def get_recent_logs(limit: int = 100):
    if not os.path.exists(monitor.log_file):
        return {"logs": []}

    with open(monitor.log_file, "r") as f:
        lines = f.readlines()[-limit:]
    return {"logs": [line.strip() for line in lines]}