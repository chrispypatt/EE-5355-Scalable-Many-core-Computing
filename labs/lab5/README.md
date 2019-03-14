# Lab 5: BFS Queueing
The purpose of this lab is to understand hierarchical queuing in the context of the breadth first search algorithm as an example. You will implement a single iteration of breadth first search that takes a set of nodes in the current level (also called wave-front) and a input and outputs the set of nodes belonging to the next level.

## Running code:
```
make
./bfs-queuing <m> 			# Mode: m, Nodes: 200,000, Max neighbors  per  node :  10
./bfs-queuing <m> <N>		# Mode: m, Nodes: N, Max neighbors  per  node :  10
./bfs-queuing <m> <N> <M>	# Mode: m, Nodes: N, Max neighbors  per  node :  M
```

There are 5 modes available:
- Mode 1 executes the CPU version
- Mode 2 executes the GPU version using just a global queue
- Mode 3 executes the GPU version with hierarchical queuing with block and global queuing
- Mode 4 executes the GPU version with hierarchical queuing with warp, block, and global queuing


