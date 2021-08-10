class SegmentTree:

    def __init__(self, arr):
        self.arr = arr
        self.arr_size = len(arr)
        self.segment_tree = [0] * (4 * self.arr_size)
        self.segment_tree_size = len(self.segment_tree)
        self._build_tree(1, 0, self.arr_size - 1)

    def _build_tree(self, index, start, end):
        if start == end:

            if index < self.segment_tree_size:
                self.segment_tree[index] = self.arr[start]

            return

        mid = (start + end) // 2

        self._build_tree(2 * index, start, mid)
        self._build_tree(2 * index + 1, mid + 1, end)
        self.segment_tree[index] = self.segment_tree[2 * index] + self.segment_tree[(2 * index) + 1]

    def _update_segment_tree(self, start, end, treeNode, index, value):
        if start == end:
            self.arr[index] = value
            self.segment_tree[treeNode] = value
            return

        mid = (end + start) // 2
        if mid > index:
            self._update_segment_tree(mid + 1, end, 2 * treeNode + 2, index, value)
        else:
            self._update_segment_tree(start, mid, 2 * treeNode, index, value)

        self.segment_tree[treeNode] = self.segment_tree[2 * treeNode] + self.segment_tree[2 * treeNode + 1]

    def update_segment_tree(self, index, value):
        self._update_segment_tree(0, self.arr_size - 1, 1, index, value)

    def print(self):
        print(s.segment_tree)

    def _query_segment_tree(self, start, end, treeNode, start_range, end_range):
        if start > end_range or end < start_range:
            return 0

        if start >= start_range and end <= end_range:
            return self.segment_tree[treeNode]

        mid = (start + end) // 2

        left = self._query_segment_tree(start, mid, 2 * treeNode, start_range, end_range)
        right = self._query_segment_tree(mid + 1, end, 2 * treeNode + 1, start_range, end_range)

        return left + right

    def query_segment_tree(self, start_range, end_range):
        return self._query_segment_tree(0, len(self.arr) - 1, 1, start_range, end_range)


s = SegmentTree([1, 2, 3, 4, 5, 6])
s.print()
s.update_segment_tree(4, 10)
s.print()
print(s.query_segment_tree(1, 4))
