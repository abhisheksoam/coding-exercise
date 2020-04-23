class Sort:
    def insertion_sort(self, input):
        """
        Insertion Sort:
        Pick an element(e1) and start iterating backwards to compare with elements to swap
        and adjust element e1 position.
        """
        for i in range(0, len(input)):
            x = input[i]
            j = i - 1
            while j >= 0 and input[j] > x:
                input[j + 1] = input[j]
                j = j - 1

            input[j + 1] = x

    def merge_sort(self, input):
        if len(input) == 1:
            return input

        mid = len(input) // 2
        left = input[0:mid]
        right = input[mid:]

        left = self.merge_sort(left)
        right = self.merge_sort(right)
        return self.merge(left, right)

    def merge(self, left, right):
        output = []
        l_index, r_index = 0, 0
        while l_index < len(left) and r_index < len(right):
            if left[l_index] < right[r_index]:
                output.append(left[l_index])
                l_index += 1
            else:
                output.append(right[r_index])
                r_index += 1

        while l_index < len(left):
            output.append(left[l_index])
            l_index += 1

        while r_index < len(right):
            output.append(right[r_index])
            r_index += 1

        return output

    def quick_sort(self, input):
        if not input:
            return input

        pivot_element = input[0]
        left = []
        right = []
        for i in range(1, len(input)):
            if input[i] <= pivot_element:
                left.append(input[i])
            else:
                right.append(input[i])

        left = self.quick_sort(left)
        right = self.quick_sort(right)
        return left + [pivot_element] + right


input = [8, 2, 4, 9, 3, 6]
print(input)
s = Sort()
# s.insertion_sort(input)
# output = s.merge_sort(input)
output = s.quick_sort(input)
print(output)
