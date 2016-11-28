import numpy as np


def checkIntersections(line1, line2):
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

if __name__ == '__main__':
    test_case1 = np.array([[[4.0, 2.0], [7.0, 9.0]],
                          [[7.0, 5.0], [9.0, 2.0]]])

    test_case2 = np.array([[[4.0, 2.0], [5.0, 7.0]],
                          [[2.0, 5.0], [7.0, 5.0]]])

    test_case3 = np.array([[[7.0, 9.0], [4.0, 12.0]],
                           [[2.0, 8.0], [7.0, 11.0]]])

    test_case4 = np.array([[[0.0, 0.0], [0.0, 12.0]],
                           [[2.0, 0.0], [2.0, 11.0]]])

    test_case5 = np.array([[[0.0, 5.0], [5.0, 5.0]],
                           [[0.0, 4.0], [6.0, 4.0]]])

    test_case6 = np.array([[[4.0, 2.0], [6.0, 3.0]],
                           [[9.0, 2.0], [7.0, 5.0]]])

    print "Result of test case #1: ",
    print checkIntersections(test_case1[0], test_case1[1])
    print "Result of test case #2: ",
    print checkIntersections(test_case2[0], test_case2[1])
    print "Result of test case #3: ",
    print checkIntersections(test_case3[0], test_case3[1])
    print "Result of test case #4: ",
    print checkIntersections(test_case4[0], test_case4[1])
    print "Result of test case #5: ",
    print checkIntersections(test_case5[0], test_case5[1])
    print "Result of test case #6: ",
    print checkIntersections(test_case6[0], test_case6[1])
