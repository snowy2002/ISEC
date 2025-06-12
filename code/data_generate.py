from cyaron import *


## 生成基本参数
io = IO(file_prefix="code/data/base")
fm = uniform(4, 8) # 服务器主频
slot = randint(8, 12) # 时隙
num_m = 10 # 服务器数量
rm = uniform(50, 100) # 传输速率
io.input_writeln(fm + 1.4133, slot, num_m, rm) # 写文件


## 生成DAG相关参数
io = IO(file_prefix="code/data/Graph")
vertex_num = randint(40, 40) # 节点数####
edge_num = int(vertex_num * uniform(2, 3))
io.input_writeln(vertex_num, edge_num)
graph = Graph.DAG(vertex_num - 1, edge_num, repeated_edges=False)
for i in range(1, vertex_num):
    graph.add_edge(i, vertex_num)
    graph.add_edge(0, i)
io.input_writeln(graph.to_str(output=Edge.unweighted_edge))


## 生成任务相关参数
io = IO(file_prefix="code/data/task")
cost = [uniform(4, 32) for i in range(vertex_num + 1)] # 任务执行时间
cost[0] = 0 
cost[vertex_num] = 0 # 起点和终点任务执行时间为0
io.input_writeln(cost)