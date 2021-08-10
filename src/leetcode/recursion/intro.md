
You might wonder how we can implement a function that calls itself. 
The trick is that each time a recursive function calls itself, 
it reduces the given problem into subproblems. 
The recursion call continues until it reaches a point where the 
subproblem can be solved without further recursion.

A recursive function should have the following properties so that it does not result in an infinite loop:

    A simple base case (or cases) — a terminating scenario that does not use recursion to produce an answer.
    A set of rules, also known as recurrence relation that reduces all other cases towards the base case.

Note that there could be multiple places where the function may call itself.

For a problem, if there exists a recursive solution, we can follow the guidelines below to implement it. 

For instance, we define the problem as the function F(X) to implement, 
where X is the input of the function which also defines the scope of the problem.

Then, in the function F(X), we will:

    Break the problem down into smaller scopes, such as x0∈X,x1∈X,...,xn∈X{x_0} \in X, {x_1} \in X, ..., {x_n} \in Xx0​∈X,x1​∈X,...,xn​∈X;
    Call function F(x0),F(x1),...,F(xn){F(x_0)}, F(x_1), ..., F(x_n)F(x0​),F(x1​),...,F(xn​) recursively to solve the subproblems of X{X}X;
    Finally, process the results from the recursive function calls to solve the problem corresponding to X
    

**Recurence Relation**    
    
There are two important things that one needs to figure out before implementing a recursive function:

    recurrence relation: the relationship between the result of a problem and the result of its subproblems.
    base case: the case where one can compute the answer directly without any further recursion calls. 
               Sometimes, the base cases are also called bottom cases, since they are often the cases where the problem has been reduced to the minimal scale, i.e. the bottom, if we consider that dividing the problem into subproblems is in a top-down manner.

  Once we figure out the above two elements, to implement a recursive function we simply call the function itself according to the recurrence relation until we reach the base case.


**Time Complexity - Recursion** 

In this article, we will focus on how to calculate the time complexity for recursion algorithms.

    Given a recursion algorithm, its time complexity O(T) 
    is typically the product of the number of recursion invocations (denoted as R) and the time complexity of 
    calculation (denoted as O(s)) that incurs along with each recursion call:

    O(T)=R∗O(s)

Let's take a look at some examples below.


As you might recall, in the problem of printReverse, 
we are asked to print the string in the reverse order. 
A recurrence relation to solve the problem can be expressed as follows:

    printReverse(str) = printReverse(str[1...n]) + print(str[0])
    
where str[1...n] is the substring of the input string str, without the leading character str[0].

As you can see, the function would be recursively invoked n times, where n is the size of the input string. At the end of each recursion, we simply print the leading character, 
therefore the time complexity of this particular operation is constant, i.e. O(1).

To sum up, the overall time complexity of our recursive function printReverse(str) would be

    O(printReverse)=n∗O(1)=O(n).
 
 
 **Execution Tree**

For recursive functions, it is rarely the case that the number of recursion calls
happens to be linear to the size of input. 
For example, one might recall the example of Fibonacci number that we discussed in 
the previous chapter, whose recurrence relation is defined as
    
    f(n) = f(n-1) + f(n-2). 
    
At first glance, it does not seem straightforward to calculate the number of 
recursion invocations during the execution of the Fibonacci function.

In this case, it is better resort to the execution tree, which is a tree that is used to denote the execution flow of a recursive function in particular. Each node in the tree represents an invocation of the recursive function. Therefore, the total number of nodes in the tree corresponds to the number of recursion calls during the execution.

The execution tree of a recursive function would form an n-ary tree, with n as the number of times recursion appears in the recurrence relation. For instance, the execution of the Fibonacci function would form a binary tree, as one can see from the following graph which shows the execution tree for the calculation of Fibonacci number f(4).

In a full binary tree with n levels, the total number of nodes would be 2n−12^n - 12n−1. Therefore, the upper bound (though not tight) for the number of recursion in f(n) would be 2n−1{2^n -1}2n−1, as well. As a result, we can estimate that the time complexity for f(n) would be O(2^n).
