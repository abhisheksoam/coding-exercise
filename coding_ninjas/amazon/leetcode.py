from typing import List


class Solution:
    def kClosest(points: List[List[int]], K: int) -> List[List[int]]:
        import math
        import heapq

        def euclidean_distance(point):
            return math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2))

        processing_dict = {}
        output = []
        for point in points:
            distance = euclidean_distance(point)
            if distance in processing_dict:
                point_list = processing_dict.get(distance)
                point_list.append(point)
            else:
                point_list = [point]

            processing_dict[distance] = point_list
            heapq.heappush(output, distance)

        result = []
        for distance in heapq.nsmallest(K, output):
            points_list = processing_dict.get(distance, None)
            if points_list is not None:
                for _point in points_list:
                    result.append(_point)
                del processing_dict[distance]

        return result

    """
    https://leetcode.com/problems/integer-to-english-words/
    Category: Hard

    """

    def numberToWords(self, num: int) -> str:
        from collections import OrderedDict
        translation_map = OrderedDict()

        translation_map[1] = "one"
        translation_map[2] = "two"
        translation_map[1] = "three"
        translation_map[1] = "four"
        translation_map[1] = "five"
        translation_map[1] = "one"
        translation_map[1] = "one"
        translation_map[1] = "one"
        translation_map[1] = "one"
        translation_map[1] = "one"
        translation_map[1] = "one"


