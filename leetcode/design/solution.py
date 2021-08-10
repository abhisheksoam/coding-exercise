from collections import OrderedDict
class LRUCache:


    def __init__(self, capacity: int):
        self.hashmap = {}
        self.capacity = capacity
        self.items_count = 0

    def get(self, key: int) -> int:
        if key in self.hashmap:
            ## increment the counter for this key

            return self.hashmap[key]

        return -1

    def put(self, key: int, value: int) -> None:
        if self.items_count == self.capacity:
            # Get the least recently used key, and update it in place of that
            pass
        else:
            self.hashmap[key] = value
            self.items_count = self.items_count + 1
