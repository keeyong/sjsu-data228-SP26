import ray
import time

ray.init(include_dashboard=True)

@ray.remote
def work(x):
    time.sleep(2)
    return x * x

start = time.time()

futures = [work.remote(i) for i in range(5)]
results = ray.get(futures)

end = time.time()

print("Results:", results)
print("Time taken:", round(end - start, 2), "seconds")

time.sleep(100)
