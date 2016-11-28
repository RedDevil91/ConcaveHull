import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, cKDTree
from matplotlib.patches import Polygon, Circle


class ConcaveHull(object):
    def __init__(self, points, k=3):
        self.points = points
        self.temp_list = np.copy(self.points)
        self.k = k
        self.vertices = list()

        self.start_point = self.findStartPoint()
        self.addPointToVertices(self.start_point)
        self.computeVertices()
        return

    def computeVertices(self):
        i = 0
        while True:
            if i > 20:
                break
            dists, idxs = self.findKNearestNeighbour(self.points[self.vertices[-1]])

            next_points, indexes = self.findNextPoint(idxs, self.getBasePoints())

            next_point = self.findIntersections(next_points)

            if (next_point[0] == self.start_point[0] and
                    next_point[1] == self.start_point[1]):
                self.addPointToVertices(self.start_point)
                break

            self.addPointToVertices(next_point)

            if len(self.vertices) == 4:
                self.temp_list = np.append(self.temp_list, [self.start_point], 0)

            i += 1
        return

    def findStartPoint(self):
        min_value = np.min(self.points[:, 1])
        return self.points[np.where(self.points[:, 1] == min_value)][0]

    def findKNearestNeighbour(self, point):
        self.temp_list = self.excludePointFromList(point, self.temp_list)
        if len(self.temp_list) <= self.k:
            # TODO: if len <= k then return the remaining points (dist, idxs)
            pass
        tree = cKDTree(self.temp_list.copy())
        return tree.query(point, k=self.k)

    def findNextPoint(self, indexes, base_points):
        base_vector = self.getVectorFromTwoPoint(base_points[0], base_points[1])
        angles = np.array([])
        points = self.temp_list[indexes]
        for point in points:
            vector = self.getVectorFromTwoPoint(base_points[0], point)
            angle = np.arctan2(np.cross(base_vector, vector), np.dot(base_vector, vector))
            angle = angle if angle < 0 else (-np.pi - (np.pi - angle))
            angles = np.append(angles, angle)
        idxs = np.argsort(angles)
        return points[idxs], indexes[idxs]

    def findIndexOfPoint(self, point):
        return np.where(np.logical_and(self.points[:, 0] == point[0],
                                       self.points[:, 1] == point[1]))[0][0]

    def addPointToVertices(self, point):
        idx = self.findIndexOfPoint(point)
        self.vertices.append(idx)
        return

    def findIntersections(self, point_list):
        if len(self.vertices) < 3:
            return point_list[0]
        for point in point_list:
            curr_segment = [self.points[self.vertices[-1]], point]
            intersects = False
            for i in xrange(len(self.vertices) - 1):
                segment = [self.points[self.vertices[i]], self.points[self.vertices[i + 1]]]
                if (self.checkSegmentIntersections(curr_segment, segment) and
                        not self.checkSamePointsInLines(curr_segment, segment)):
                    intersects = True
                    break
            if not intersects:
                return point

    def getBasePoints(self):
        if len(self.vertices) == 1:
            base_points = [self.start_point, np.array([self.start_point[0] - 1, self.start_point[1]])]
        else:
            base_points = [self.points[self.vertices[-1]], self.points[self.vertices[-2]]]
        return base_points

    @staticmethod
    def getVectorFromTwoPoint(start_point, end_point):
        return np.array([(end_point[0] - start_point[0]), (end_point[1] - start_point[1])])

    @staticmethod
    def excludePointFromList(point, point_list):
        return np.delete(point_list, np.where(np.logical_and(point_list[:, 1] == point[1],
                                                             point_list[:, 0] == point[0])),
                         axis=0)

    @staticmethod
    def checkSegmentIntersections(line1, line2):
        r = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
        s = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
        cross_prod = np.cross(r, s)
        if cross_prod == 0:
            return False
        diff = (line2[0][0] - line1[0][0], line2[0][1] - line1[0][1])
        t = np.cross(diff, s) / cross_prod
        u = np.cross(diff, r) / cross_prod
        if (0 <= t <= 1) and (0 <= u <= 1):
            return True
        return False

    @staticmethod
    def checkSamePointsInLines(line1, line2):
        return (line1[0][0] == line2[0][0] or line1[0][0] == line2[1][0] or
                line1[1][0] == line2[0][0] or line1[1][0] == line2[1][0] or
                line1[0][1] == line2[0][1] or line1[0][1] == line2[1][1] or
                line1[1][1] == line2[0][1] or line1[1][1] == line2[1][1])

if __name__ == '__main__':
    # np.random.seed(3)
    # np.random.seed(2)
    np.random.seed(4)
    points = np.random.random_sample((30, 2))
    # points = np.array([
    #     [7, 1],
    #     [4, 2],
    #     [9, 2],
    #     [6, 3],
    #     [3, 4],
    #     [2, 5],
    #     [4, 5],
    #     [7, 5],
    #     [5, 7],
    #     [2, 8],
    #     [4, 8],
    #     [7, 9],
    #     [3, 10],
    #     [5, 10],
    #     [6, 11],
    #     [4, 12],
    # ])

    plt.plot(points[:, 0], points[:, 1], 'o', ms=10, color='blue')

    hull = ConcaveHull(points, k=6)
    plt.plot(hull.start_point[0], hull.start_point[1], 'o', ms=10, color='red')

    # dist, idxs = hull.findKNearestNeighbour(hull.start_point)
    # plt.plot(hull.temp_list[idxs, 0], hull.temp_list[idxs, 1], 'o', ms=10, color='green')
    #
    # ax = plt.subplot()
    #
    # ax.add_patch(Circle(hull.start_point, radius=np.max(dist), fill=False))
    #
    # temp_point = np.array([(hull.start_point[0] - 1), hull.start_point[1]])
    # next_point, indexes = hull.findNextPoint(idxs, [hull.start_point, temp_point])
    # plt.plot(next_point[0][0], next_point[0][1], 'o', ms=10, color='yellow')
    #
    # hull.addPointToVertices(next_point[0])
    #
    # dist, idxs = hull.findKNearestNeighbour(next_point[0])
    # plt.plot(hull.temp_list[idxs, 0], hull.temp_list[idxs, 1], 'o', ms=10, color='green')
    #
    # next_point2, indexes = hull.findNextPoint(idxs, [next_point[0], hull.start_point])
    # plt.plot(next_point2[0][0], next_point2[0][1], 'o', ms=10, color='yellow')
    #
    # hull.addPointToVertices(next_point2[0])
    #
    # dist, idxs = hull.findKNearestNeighbour(next_point2[0])
    # plt.plot(hull.temp_list[idxs, 0], hull.temp_list[idxs, 1], 'o', ms=10, color='green')
    #
    # next_point3, indexes = hull.findNextPoint(idxs, [next_point2[0], next_point[0]])
    # plt.plot(next_point3[0][0], next_point3[0][1], 'o', ms=10, color='yellow')
    #
    # hull.addPointToVertices(next_point3[0])

    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], '-')
    plt.show()
