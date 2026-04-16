from pyflink.datastream import StreamExecutionEnvironment 
from pyflink.common import Types 

# 1. Set up the streaming execution environment 
env = StreamExecutionEnvironment.get_execution_environment() 
env.set_parallelism(2) 

# 2. Build a source stream from an in-memory collection 
lines = env.from_collection([ 
    "hello world hello flink", 
    "flink streaming wordcount example", 
], type_info=Types.STRING()) 

# 3. split → tuple(word,1) → keyBy word → sum counts 
counts = (lines 
    .flat_map(lambda line: [(w, 1) for w in line.split()], 
             output_type=Types.TUPLE([Types.STRING(), Types.INT()])) 
    .key_by(lambda x: x[0]) 
    .sum(1)) 

# 4. Sink to stdout and trigger execution 
counts.print() 
env.execute("WordCount")

