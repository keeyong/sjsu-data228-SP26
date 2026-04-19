import time

def work(x):
    time.sleep(2)
    return x * x

start = time.time()

results = []
for i in range(5):
    results.append(work(i))

end = time.time()

print("Results:", results)
print("Time taken:", round(end - start, 2), "seconds")
