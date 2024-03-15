import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from dask.distributed import Client
from distributed.diagnostics import MemorySampler

class MemoryAnalysis:
    def __init__(self, logmemoryinfo, use_mp, csvfilename="", pngfilename=""):
        self.logmemoryinfo = logmemoryinfo

        if self.logmemoryinfo:
            self.use_mp = use_mp
            self.csvfilename = csvfilename
            self.pngfilename = pngfilename
            self.memory_df = None
            self.memory_sampler = None
            self.start_time = None
            self.start_memory = None

            if not use_mp:
                self.memory_sampler = MemorySampler()
            else:
                self.context_manager = None
                self.memory_df = pd.DataFrame(columns=["Name", "StartTime", "EndTime", "StartMemory", "EndMemory"])

    def sample(self, name):
        self.name = name
        return self
            
    def __enter__(self):
        if self.logmemoryinfo:
            if self.use_mp:
                self.start_time = time.time()
                self.start_memory = psutil.virtual_memory().used / (1024**3)
            else:
                self.context_manager = self.memory_sampler.sample(self.name)
                self.context_manager.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logmemoryinfo:
            if self.use_mp:
                end_time = time.time()

                end_memory = psutil.virtual_memory().used / (1024**3)

                self.memory_df = pd.concat([self.memory_df, pd.DataFrame([{"Name": self.name, "StartTime": self.start_time, "EndTime": end_time, "StartMemory": self.start_memory, "EndMemory": end_memory}])], ignore_index=True)

                self.memory_df.to_csv(self.csvfilename, index=False)    
            else:
                self.context_manager.__exit__(exc_type, exc_val, exc_tb)

    def log(self):
        if self.logmemoryinfo:
            if self.use_mp:
                fig = self.plot(figsize=(12, 6))
                fig.savefig(self.pngfilename)
            else:
                fig = self.memory_sampler.plot(align=True, figsize=(12, 6))
                plt.savefig(self.pngfilename)

    def plot(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        for _, row in self.memory_df.iterrows():
            plt.plot([row["StartTime"], row["EndTime"]], [row["StartMemory"], row["EndMemory"]], marker="o", label=row["Name"])

        plt.title("Memory Usage Over Time")
        plt.xlabel("time")
        plt.ylabel("Memory Usage (GiB)")

        plt.xticks(rotation=45)

        plt.legend(title="Task Name")

        plt.tight_layout()

        plt.show()

        return fig