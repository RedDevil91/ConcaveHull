import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree
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
            if i > 5:
                break
            dists, idxs = self.findKNearestNeighbour(self.points[self.vertices[-1]])
            if len(self.vertices) == 1:
                base_points = [self.start_point, np.array([self.start_point[0]-1,self.start_point[1]])]
            else:
                base_points = [self.points[self.vertices[-1]], self.points[self.vertices[-2]]]

            next_point, index = self.findNextPoint(idxs, base_points)
            if self.checkIntersections(next_point) and i > 3:
                idxs = np.delete(idxs, np.where(idxs == index))
            self.addPointToVertices(next_point)
            i += 1
        return

    def findStartPoint(self):
        min_value = np.min(self.points[:, 1])
        return self.points[np.where(self.points[:, 1] == min_value)][0]

    def findKNearestNeighbour(self, point):
        self.temp_list = self.excludePointFromList(point, self.temp_list)
        tree = KDTree(self.temp_list.copy())
        return tree.query(point, k=self.k)

    def findNextPoint(self, indexes, base_points):
        base_vector = self.getVectorFromTwoPoint(base_points[0], base_points[1])
        min_angle_point = None
        index = 0
        min_angle = 4.0
        for idx in indexes:
            point = self.temp_list[idx]
            vector = self.getVectorFromTwoPoint(base_points[0], point)
            angle = np.arctan2(np.cross(base_vector, vector), np.dot(base_vector, vector))
            if angle < min_angle:
                min_angle_point = point
                min_angle = angle
                index = idx
        return min_angle_point, index

    def findIndexOfPoint(self, point):
        return np.where(np.logical_and(self.points[:, 0] == point[0],
                                       self.points[:, 1] == point[1]))[0][0]

    def addPointToVertices(self, point):
        idx = self.findIndexOfPoint(point)
        self.vertices.append(idx)
        return

    def checkIntersections(self, point):
        curr_segment = [self.points[self.vertices[-1]], point]
        r = (curr_segment[1][0] - curr_segment[0][0], curr_segment[1][1] - curr_segment[0][1])
        for i in range(len(self.vertices)-1):
            segment = [self.points[self.vertices[i]], self.points[self.vertices[i+1]]]
            if not self.checkBoundingBoxIntersect(segment, curr_segment):
                continue
            s = (segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])
            cross_prod = np.cross(r, s)
            if cross_prod == 0:
                continue
            diff = (segment[0][0] - curr_segment[0][0], segment[0][1] - curr_segment[0][1])
            t = np.cross(diff, s) / cross_prod
            u = np.cross(diff, r) / cross_prod
            if (0 <= t <= 1 and 0 <= u <= 1):
                print point
                return True
        return False

    @staticmethod
    def getVectorFromTwoPoint(startPoint, endPoint):
        return np.array([(endPoint[0] - startPoint[0]), (endPoint[1] - startPoint[1])])

    @staticmethod
    def excludePointFromList(point, point_list):
        return np.delete(point_list, np.where(np.logical_and(point_list[:, 1] == point[1],
                                                             point_list[:, 0] == point[0])),
                         axis=0)

    @staticmethod
    def checkBoundingBoxIntersect(box1, box2):
        return (box1[0][0] <= box2[1][0] and box1[1][0] >= box2[0][0] and
                box1[0][1] <= box2[1][1] and box1[1][1] >= box2[0][1])


if __name__ == '__main__':
    points = np.random.random_sample((10, 2))
    plt.plot(points[:, 0], points[:, 1], 'o', ms=10, color='blue')

    hull = ConcaveHull(points, k=3)
    plt.plot(hull.start_point[0], hull.start_point[1], 'o', ms=10, color='red')

    # dist, idxs = hull.findKNearestNeighbour(hull.start_point)
    # plt.plot(hull.temp_list[idxs, 0], hull.temp_list[idxs, 1], 'o', ms=10, color='green')
    #
    # ax = plt.subplot()
    #
    # ax.add_patch(Circle(hull.start_point, radius=np.max(dist), fill=False))
    #
    # temp_point = np.array([(hull.start_point[0] - 1), hull.start_point[1]])
    # next_point = hull.findNextPoint(idxs, [hull.start_point, temp_point])
    # plt.plot(next_point[0], next_point[1], 'o', ms=10, color='yellow')
    #
    # hull.addPointToVertices(next_point)
    #
    # dist, idxs = hull.findKNearestNeighbour(next_point)
    # plt.plot(hull.temp_list[idxs, 0], hull.temp_list[idxs, 1], 'o', ms=10, color='green')
    #
    # next_point2 = hull.findNextPoint(idxs, [next_point, hull.start_point])
    # plt.plot(next_point2[0], next_point2[1], 'o', ms=10, color='yellow')

    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], '-')
    plt.show()