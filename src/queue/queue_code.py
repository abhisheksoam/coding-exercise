class Queue:
    def __init__(self):
        self.qlist = []
        self.head = -1
        self.tail = -1

    def enqueue(self, item):
        if self.head == -1:
            self.head += 1
        self.tail += 1
        self.qlist.append(item)

    def dequeue(self):
        if self.head == -1:
            print("queue underflow")
        elif self.head == self.tail:
            p = self.qlist.pop(0)
            self.head -= 1
            self.tail -= 1
            return p
        else:
            self.tail -= 1
            return self.qlist.pop(0)

    def size(self):
        return len(self.qlist)

    def peek(self):
        if self.head == -1:
            print("queue empty")
        else:
            return self.qlist[self.head]
