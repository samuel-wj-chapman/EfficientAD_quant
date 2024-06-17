import psutil
import time

def monitor_resources(interval=1):
    # Get the current process
    current_process = psutil.Process()

    while True:
        # Number of threads
        num_threads = current_process.num_threads()

        # RAM usage
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB

        print(f"Threads: {num_threads}, RAM Usage: {ram_usage:.2f} GB")

        # Sleep for the specified interval
        time.sleep(interval)

if __name__ == "__main__":
    monitor_resources()