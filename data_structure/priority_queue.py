"""
A queue is termed as a priority queue if it has the following characteristics:

- Each item has some priority associated with it.
- An item with the highest priority is moved at the front and deleted first.
- If two elements share the same priority value, then the priority queue follows the
  first-in-first-out principle for de queue operation.

A priority queue is of two types:

    Ascending Order Priority Queue
    Descending Order Priority Queue

A typical priority queue supports following operations.

insert(item, priority): Inserts an item with given priority.
getHighestPriority(): Returns the highest priority item.
deleteHighestPriority(): Removes the highest priority item.


You can implement the priority queues in one of the following ways:

    Linked list
    Binary heap
    Arrays
    Binary search tree
With Fibonacci heap, insert() and getHighestPriority()
can be implemented in O(1) amortized time and deleteHighestPriority() can be implemented in O(Logn) amortized time.

Operation 	Unordered Array 	Ordered Array 	Binary Heap 	Binary Search Tree
Insert  	0(1) 	                0(N) 	      0(log(N)) 	0(log(N))
Peek 	    0(N) 	                0(1) 	      0(1) 	        0(1)
Delete 	    0(N) 	                0(1) 	      0(log (N)) 	0(log(N))


Applications of Priority Queue:
1) CPU Scheduling
2) Graph algorithms like Dijkstra’s shortest path algorithm, Prim’s Minimum Spanning Tree, etc
3) All queue applications where priority is involved.

"""

import heapq