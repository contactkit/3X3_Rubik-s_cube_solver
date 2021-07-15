import numpy as np
import cv2
import random


def shuffle():
    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), -1)

    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), -1)
    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), 1)

    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), -1)
    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), -1)

    for t in range(1):
        rotate(random.randrange(0, 5), 1)
    for t in range(1):
        rotate(random.randrange(0, 5), -1)

global cntr
cntr = int(-1)
global stepa
stepa= np.empty(shape=200, dtype='object')


global steps
steps = 0


def cor2face(face, I, J, K, f):
    # print(I, J, K)
    if f[0] == 1:
        face = 1
    elif f[0] == -1:
        face = 4
    elif f[1] == 1:
        face = 2
    elif f[1] == -1:
        face = 3
    elif f[2] == 1:
        face = 5
    elif f[2] == -1:
        face = 0

    if I == 0:
        if J == 0:
            if K == 1:
                return [5, 0]
            elif K == -1:
                return [0, 0]
        elif J == 1:
            if K == 1:
                if face == 2:
                    return [2, 1]
                elif face == 5:
                    return [5, 2]
            elif K == -1:
                if face == 2:
                    return [2, 3]
                elif face == 0:
                    return [0, 2]
            elif K == 0:
                return [2, 0]
        elif J == -1:
            if K == 1:
                if face == 3:
                    return [3, 1]
                elif face == 5:
                    return [5, 4]
            elif K == -1:
                if face == 3:
                    return [3, 3]
                elif face == 0:
                    return [0, 4]
            elif K == 0:
                return [3, 0]

    if I == 1:
        if J == 0:
            if K == 1:
                if face == 1:
                    return [1, 1]
                elif face == 5:
                    return [5, 3]
            elif K == -1:
                if face == 1:
                    return [1, 3]
                elif face == 0:
                    return [0, 1]
            elif K == 0:
                return [1, 0]
        elif J == 1:
            if K == 1:
                if face == 2:
                    return [2, 8]
                elif face == 5:
                    return [5, 6]
                elif face == 1:
                    return [1, 5]
            elif K == -1:
                if face == 2:
                    return [2, 7]
                elif face == 0:
                    return [0, 5]
                elif face == 1:
                    return [1, 6]
            elif K == 0:
                if face == 1:
                    return [1, 2]
                elif face == 2:
                    return [2, 4]
        elif J == -1:
            if K == 1:
                if face == 3:
                    return [3, 5]
                elif face == 5:
                    return [5, 7]
                elif face == 1:
                    return [1, 8]
            elif K == -1:
                if face == 3:
                    return [3, 6]
                elif face == 0:
                    return [0, 8]
                elif face == 1:
                    return [1, 7]
            elif K == 0:
                if face == 1:
                    return [1, 4]
                elif face == 3:
                    return [3, 2]

    if I == -1:
        if J == 0:
            if K == 1:
                if face == 4:
                    return [4, 1]
                elif face == 5:
                    return [5, 1]
            elif K == -1:
                if face == 4:
                    return [4, 3]
                elif face == 0:
                    return [0, 3]
            elif K == 0:
                return [4, 0]
        elif J == 1:
            if K == 1:
                if face == 2:
                    return [2, 5]
                elif face == 5:
                    return [5, 5]
                elif face == 4:
                    return [4, 8]
            elif K == -1:
                if face == 2:
                    return [2, 6]
                elif face == 0:
                    return [0, 6]
                elif face == 4:
                    return [4, 7]
            elif K == 0:
                if face == 4:
                    return [4, 4]
                elif face == 2:
                    return [2, 2]
        elif J == -1:
            if K == 1:
                if face == 3:
                    return [3, 8]
                elif face == 5:
                    return [5, 8]
                elif face == 4:
                    return [4, 5]
            elif K == -1:
                if face == 3:
                    return [3, 7]
                elif face == 0:
                    return [0, 7]
                elif face == 4:
                    return [4, 6]
            elif K == 0:
                if face == 4:
                    return [4, 2]
                elif face == 3:
                    return [3, 4]
    else:
        return -1


def printfaces():
    faces = np.empty(shape=(6, 3, 3), dtype='object')
    faces[0][0] = 0
    for ix in range(6):
        for jy in range(9):
            # print(ix, ' ', jy)
            a = cor2face(ix, face_M[ix][jy][0][0], face_M[ix][jy][0][1], face_M[ix][jy][0][2], face_M[ix][jy][1])
            # print(a)
            if a[1] == 0:
                t1 = 1
                t2 = 1
            elif a[1] == 1:
                t1 = 0
                t2 = 1
            elif a[1] == 2:
                t1 = 1
                t2 = 2
            elif a[1] == 3:
                t1 = 2
                t2 = 1
            elif a[1] == 4:
                t1 = 1
                t2 = 0
            elif a[1] == 5:
                t1 = 0
                t2 = 2
            elif a[1] == 6:
                t1 = 2
                t2 = 2
            elif a[1] == 7:
                t1 = 2
                t2 = 0
            elif a[1] == 8:
                t1 = 0
                t2 = 0

            faces[a[0]][t1][t2] = ix
            # print('//')
    for l in range(6):
        print('*********************************************************************')
        print(faces[l])
    return faces


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def cross(v1, v2):
    ix = v1[1] * v2[2] - v2[1] * v1[2]
    jy = v2[0] * v1[2] - v1[0] * v2[2]
    kz = v1[0] * v2[1] - v2[0] * v1[1]
    return [ix, jy, kz]


def rotate(face, direction):
    Face=face
    if Face==0:
        Face=6
    global stepa
    global cntr
    if cntr==-1:
        stepa[0]=Face*direction
        cntr=1
    elif stepa[cntr-1]==-1*Face*direction:
        cntr-=1
    elif cntr>1 and stepa[cntr-1]==Face*direction and stepa[cntr-2]==Face*direction:
        cntr-=2
        stepa[cntr]=-1*Face*direction
        cntr+=1
    else:
        stepa[cntr]=Face*direction
        cntr+=1
    global steps
    steps += 1
    count = 0
    # ------------------RED----------------------------------------
    if face == 1:
        if direction == 1:
            print('RC')
            v3 = [1, 0, 0]
            v31 = [1, 0, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]
                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('RAC')
            v3 = [-1, 0, 0]
            v31 = [1, 0, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]
                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1

    # ------------------ORANGE----------------------------------------
    if face == 4:
        if direction == 1:
            print('OC')
            v3 = [-1, 0, 0]
            v31 = [-1, 0, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('OAC')
            v3 = [1, 0, 0]
            v31 = [-1, 0, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1

    # ------------------GREEN----------------------------------------
    if face == 2:

        if direction == 1:
            print('GC')
            v3 = [0, 1, 0]
            v31 = [0, 1, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]
                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('GAC')
            v3 = [0, -1, 0]
            v31 = [0, 1, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]
                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
    # ------------------BLUE----------------------------------------
    if face == 3:
        if direction == 1:
            print('BC')
            v3 = [0, -1, 0]
            v31 = [0, -1, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('BAC')
            v3 = [0, 1, 0]
            v31 = [0, -1, 0]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
    # ------------------YELLOW----------------------------------------
    if face == 5:
        if direction == 1:
            print('YC')
            v3 = [0, 0, 1]
            v31 = [0, 0, 1]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('YAC')
            v3 = [0, 0, -1]
            v31 = [0, 0, 1]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
    # ------------------WHITE----------------------------------------
    if face == 0:

        if direction == 1:
            print('WC')
            v3 = [0, 0, -1]
            v31 = [0, 0, -1]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v3) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v3)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v31)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1
        if direction == -1:
            print('WAC')
            v3 = [0, 0, 1]
            v31 = [0, 0, -1]
            for ix in range(6):
                for jy in range(9):
                    if dot(face_M[ix][jy][0], v31) == 1:
                        vc0 = cross(face_M[ix][jy][0], v3)
                        d0 = dot(face_M[ix][jy][0], v31)
                        vd0 = [v31[0] * d0, v31[1] * d0, v31[2] * d0]
                        face_M[ix][jy][0] = [vd0[0] + vc0[0], vd0[1] + vc0[1], vd0[2] + vc0[2]]

                        vc1 = cross(face_M[ix][jy][1], v3)
                        d1 = dot(face_M[ix][jy][1], v31)
                        vd1 = [v31[0] * d1, v31[1] * d1, v31[2] * d1]
                        face_M[ix][jy][1] = [vd1[0] + vc1[0], vd1[1] + vc1[1], vd1[2] + vc1[2]]
                        count = count + 1


def find2(cor):
    a = block[cor[0]][cor[1]][cor[2]][0]
    ac = block[cor[0]][cor[1]][cor[2]][3]
    b = block[cor[0]][cor[1]][cor[2]][1]
    bc = block[cor[0]][cor[1]][cor[2]][4]

    # ----RED----
    if a == 1 and b == 5:
        return [1, 1], ac, [5, 3], bc
    elif a == 1 and b == 2:
        return [1, 2], ac, [2, 4], bc
    elif a == 1 and b == 0:
        return [1, 3], ac, [0, 1], bc
    elif a == 1 and b == 3:
        return [1, 4], ac, [3, 2], bc

    # ----GREEN-----
    if a == 2 and b == 5:
        return [2, 1], ac, [5, 2], bc
    elif a == 2 and b == 4:
        return [2, 2], ac, [4, 4], bc
    elif a == 2 and b == 0:
        return [2, 3], ac, [0, 2], bc
    elif a == 2 and b == 1:
        return [2, 4], ac, [1, 2], bc

    #     ------ORANGE----
    if a == 4 and b == 5:
        return [4, 1], ac, [5, 1], bc
    elif a == 4 and b == 3:
        return [4, 2], ac, [3, 4], bc
    elif a == 4 and b == 0:
        return [4, 3], ac, [0, 3], bc
    elif a == 4 and b == 2:
        return [4, 4], ac, [2, 2], bc

    #     -----BLUE----
    if a == 3 and b == 5:
        return [3, 1], ac, [5, 4], bc
    elif a == 3 and b == 1:
        return [3, 2], ac, [1, 4], bc
    elif a == 3 and b == 0:
        return [3, 3], ac, [0, 4], bc
    elif a == 3 and b == 4:
        return [3, 4], ac, [4, 2], bc

    #     ---WHITE----
    if a == 0 and b == 1:
        return [0, 1], ac, [1, 3], bc
    elif a == 0 and b == 2:
        return [0, 2], ac, [2, 3], bc
    elif a == 0 and b == 4:
        return [0, 3], ac, [4, 3], bc
    elif a == 0 and b == 3:
        return [0, 4], ac, [3, 3], bc

    #     ------YELLOW---
    if a == 5 and b == 4:
        return [5, 1], ac, [4, 1], bc
    elif a == 5 and b == 2:
        return [5, 2], ac, [2, 1], bc
    elif a == 5 and b == 1:
        return [5, 3], ac, [1, 1], bc
    elif a == 5 and b == 3:
        return [5, 4], ac, [3, 1], bc


def find3(cor):
    a = block[cor[0]][cor[1]][cor[2]][0]
    ac = block[cor[0]][cor[1]][cor[2]][3]
    b = block[cor[0]][cor[1]][cor[2]][1]
    bc = block[cor[0]][cor[1]][cor[2]][4]
    c = block[cor[0]][cor[1]][cor[2]][2]
    cc = block[cor[0]][cor[1]][cor[2]][5]

    # ----RED----
    if a == 1:
        if b == 5 and c == 2:
            return [1, 5], ac, [5, 6], bc, [2, 8], cc
        elif b == 2 and c == 5:
            return [1, 5], ac, [2, 8], bc, [5, 6], cc

        if b == 2 and c == 0:
            return [1, 6], ac, [2, 7], bc, [0, 5], cc
        elif b == 0 and c == 2:
            return [1, 6], ac, [0, 5], bc, [2, 7], cc

        if b == 3 and c == 0:
            return [1, 7], ac, [3, 6], bc, [0, 8], cc
        elif b == 0 and c == 3:
            return [1, 7], ac, [0, 8], bc, [3, 6], cc

        if b == 3 and c == 5:
            return [1, 8], ac, [3, 5], bc, [5, 7], cc
        elif b == 5 and c == 3:
            return [1, 8], ac, [5, 7], bc, [3, 5], cc

    # ----GREEN-----
    if a == 2:
        if b == 5 and c == 4:
            return [2, 5], ac, [5, 5], bc, [4, 8], cc
        elif b == 4 and c == 5:
            return [2, 5], ac, [4, 8], bc, [5, 5], cc

        if b == 4 and c == 0:
            return [2, 6], ac, [4, 7], bc, [0, 6], cc
        elif b == 0 and c == 4:
            return [2, 6], ac, [0, 6], bc, [4, 7], cc

        if b == 1 and c == 0:
            return [2, 7], ac, [1, 6], bc, [0, 5], cc
        elif b == 0 and c == 1:
            return [2, 7], ac, [0, 5], bc, [1, 6], cc

        if b == 1 and c == 5:
            return [2, 8], ac, [1, 5], bc, [5, 6], cc
        elif b == 5 and c == 1:
            return [2, 8], ac, [5, 6], bc, [1, 5], cc

    #     ------ORANGE----
    if a == 4:
        if b == 5 and c == 3:
            return [4, 5], ac, [5, 8], bc, [2, 8], cc
        elif b == 3 and c == 5:
            return [4, 5], ac, [2, 8], bc, [5, 8], cc

        if b == 3 and c == 0:
            return [4, 6], ac, [3, 7], bc, [0, 7], cc
        elif b == 0 and c == 3:
            return [4, 6], ac, [0, 7], bc, [3, 7], cc

        if b == 2 and c == 0:
            return [4, 7], ac, [2, 6], bc, [0, 6], cc
        elif b == 0 and c == 2:
            return [4, 7], ac, [0, 6], bc, [2, 6], cc

        if b == 2 and c == 5:
            return [4, 8], ac, [2, 5], bc, [5, 5], cc
        elif b == 5 and c == 2:
            return [4, 8], ac, [5, 5], bc, [2, 5], cc

    #     -----BLUE----
    if a == 3:
        if b == 5 and c == 1:
            return [3, 5], ac, [5, 7], bc, [1, 8], cc
        elif b == 1 and c == 5:
            return [3, 5], ac, [1, 8], bc, [5, 7], cc

        if b == 1 and c == 0:
            return [3, 6], ac, [1, 7], bc, [0, 8], cc
        elif b == 0 and c == 1:
            return [3, 6], ac, [0, 8], bc, [1, 7], cc

        if b == 4 and c == 0:
            return [3, 7], ac, [4, 6], bc, [0, 7], cc
        elif b == 0 and c == 4:
            return [3, 7], ac, [0, 7], bc, [4, 6], cc

        if b == 4 and c == 5:
            return [3, 8], ac, [4, 5], bc, [5, 8], cc
        elif b == 5 and c == 4:
            return [3, 8], ac, [5, 8], bc, [4, 5], cc

    #     ---WHITE----
    if a == 0:
        if b == 1 and c == 2:
            return [0, 5], ac, [1, 6], bc, [2, 7], cc
        elif b == 2 and c == 1:
            return [0, 5], ac, [2, 7], bc, [1, 6], cc

        if b == 2 and c == 4:
            return [0, 6], ac, [2, 6], bc, [4, 7], cc
        elif b == 4 and c == 2:
            return [0, 6], ac, [4, 7], bc, [2, 6], cc

        if b == 3 and c == 4:
            return [0, 7], ac, [3, 6], bc, [0, 6], cc
        elif b == 4 and c == 3:
            return [0, 7], ac, [0, 6], bc, [3, 6], cc

        if b == 3 and c == 1:
            return [0, 8], ac, [3, 5], bc, [1, 7], cc
        elif b == 1 and c == 3:
            return [0, 8], ac, [1, 7], bc, [3, 5], cc

    #     ------YELLOW---
    if a == 5:
        if b == 4 and c == 2:
            return [5, 5], ac, [4, 8], bc, [2, 5], cc
        elif b == 2 and c == 4:
            return [5, 5], ac, [2, 5], bc, [4, 8], cc

        if b == 2 and c == 1:
            return [5, 6], ac, [2, 8], bc, [1, 5], cc
        elif b == 1 and c == 2:
            return [5, 6], ac, [1, 5], bc, [2, 8], cc

        if b == 3 and c == 1:
            return [5, 7], ac, [3, 5], bc, [1, 8], cc
        elif b == 1 and c == 3:
            return [5, 7], ac, [1, 8], bc, [3, 5], cc

        if b == 3 and c == 4:
            return [5, 8], ac, [3, 8], bc, [4, 5], cc
        elif b == 4 and c == 3:
            return [5, 8], ac, [4, 5], bc, [3, 8], cc


def insert(c, v, cor):
    if block[cor[0]][cor[1]][cor[2]] == None:
        block[cor[0]][cor[1]][cor[2]] = [c, -1, -1, v, [0, 0, 0], [0, 0, 0]]
    else:
        if block[cor[0]][cor[1]][cor[2]][1] == -1:
            block[cor[0]][cor[1]][cor[2]][1] = c
            block[cor[0]][cor[1]][cor[2]][4] = v
        elif block[cor[0]][cor[1]][cor[2]][2] == -1:
            block[cor[0]][cor[1]][cor[2]][2] = c
            block[cor[0]][cor[1]][cor[2]][5] = v
        else:
            print("already full")


def urldf(C1, C2):
    if (C1 == 1 and C2 == 2) or (C1 == 2 and C2 == 1):
        u = 5
        d = 0
        r = 2
        l = 3
        f = 1
    elif (C1 == 1 and C2 == 3) or (C1 == 3 and C2 == 1):
        u = 5
        d = 0
        r = 1
        l = 4
        f = 3
    elif (C1 == 4 and C2 == 2) or (C1 == 2 and C2 == 4):
        u = 5
        d = 0
        r = 4
        l = 1
        f = 2
    elif (C1 == 4 and C2 == 3) or (C1 == 3 and C2 == 4):
        u = 5
        d = 0
        r = 3
        l = 2
        f = 4
    return u, r, l, d, f


def back(f):
    b = 1
    if f == 1:
        b = 4
    elif f == 2:
        b = 3
    elif f == 3:
        b = 2
    elif f == 5:
        b = 0
    elif f == 0:
        b = 5
    return b


# *********************************F2L_CASES***********************************


#            white  up
def wtu1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtu2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtu3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, 1)
    rotate(r, -1)
    rotate(f, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtu4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtu6(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtu5(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtu8(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(f, -1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtu7(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


#         *****************white up + middle edge slot***********************
def wtue1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtue12(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(u, -1)
    rotate(r, -1)


def wtue2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(l, 1)
    rotate(f, -1)
    rotate(l, 1)
    rotate(l, 1)
    rotate(u, 1)
    rotate(l, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, 1)


def wtue22(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(f, 1)
    rotate(u, 1)
    rotate(f, 1)


def wtue3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    b = back(f)
    rotate(b, -1)
    rotate(r, 1)
    rotate(b, -1)
    rotate(b, -1)
    rotate(u, -1)
    rotate(b, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtue32(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtue4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    b = back(f)
    rotate(r, 1)
    rotate(r, 1)
    rotate(d, 1)
    rotate(b, 1)
    rotate(b, 1)
    rotate(d, -1)
    rotate(r, -1)
    rotate(r, -1)


def wtue42(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    b = back(f)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(b, -1)
    rotate(u, 1)
    rotate(b, 1)
    rotate(r, -1)


#     ******************************WHITE RIGHT*****************************
def wtr1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtr2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtr3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)


def wtr4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtr5(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtr6(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtr7(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtr8(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


# ***************************white right+ + middle edge slot************************
def wtre1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtre12(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtre2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtre22(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(l, -1)
    rotate(u, -1)
    rotate(l, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtre3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(l, 1)
    rotate(u, 1)
    rotate(l, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtre32(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    b = back(f)
    rotate(l, -1)
    rotate(b, 1)
    rotate(l, 1)
    rotate(b, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


def wtre4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def wtre42(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, -1)
    rotate(u, -1)
    rotate(r, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


#     ******************************WHITE FRONT*****************************not started
def wtf1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtf2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtf3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, -1)


def wtf4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)
    rotate(u, -1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtf5(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtf6(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtf7(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtf8(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(r, -1)
    rotate(f, 1)
    rotate(r, 1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)


# ***************************white FRONT+ + middle edge slot************************
def wtfe1(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtfe12(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, -1)
    rotate(r, 1)
    rotate(u, 1)
    rotate(r, -1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtfe2(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, -1)
    rotate(d, -1)
    rotate(f, -1)
    rotate(d, 1)
    rotate(r, 1)


def wtfe22(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(f, 1)
    rotate(u, 1)
    rotate(f, -1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtfe3(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(u, 1)
    rotate(l, 1)
    rotate(u, -1)
    rotate(l, -1)
    rotate(r, 1)
    rotate(u, -1)
    rotate(r, -1)


def wtfe32(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(l, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)
    rotate(l, -1)


def wtfe4(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    rotate(r, -1)
    rotate(u, 1)
    rotate(u, 1)
    rotate(r, 1)
    rotate(f, -1)
    rotate(u, -1)
    rotate(f, 1)


def wtfe42(C1, C2):
    u, r, l, d, f = urldf(C1, C2)
    b = back(f)
    rotate(u, 1)
    rotate(b, 1)
    rotate(u, 1)
    rotate(b, -1)
    rotate(f, -1)
    rotate(u, 1)
    rotate(f, 1)


def f2l():
    # **************RED_GREEN***top to position*********************
    if (face_M[0][5][0] == [1, 1, -1] and face_M[0][5][1] == [0, 0, -1] and face_M[1][2][0] == [1, 1, 0] and
            face_M[1][2][1] == [1, 0, 0]):
        face_M[0][5][0] = [1, 1, -1]
    elif (face_M[0][5][0] == [1, 1, -1]):
        rotate(1, -1)
        rotate(5, -1)
        rotate(1, 1)
        rotate(5, 1)
    elif (face_M[0][5][0] == [1, -1, -1]):
        rotate(3, -1)
        rotate(5, -1)
        rotate(3, 1)
    elif (face_M[0][5][0] == [-1, -1, -1]):
        rotate(4, -1)
        rotate(5, -1)
        rotate(4, 1)
        rotate(5, -1)
    elif (face_M[0][5][0] == [-1, 1, -1]):
        rotate(4, 1)
        rotate(5, 1)
        rotate(4, -1)
    elif (face_M[0][5][0] == [1, -1, 1]):
        rotate(5, -1)
    elif (face_M[0][5][0] == [-1, -1, 1]):
        rotate(5, -1)
        rotate(5, -1)
    elif (face_M[0][5][0] == [-1, 1, 1]):
        rotate(5, 1)

    #     **************red_green f2l********************************************************************************************

    # white up****************

    if (face_M[0][5][1] == [0, 0, 1]):
        if (face_M[1][2][0] == [1, 1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtue1(1, 2)
            else:
                wtue12(1, 2)
        elif (face_M[1][2][0] == [1, -1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtue22(1, 2)
            else:
                wtue2(1, 2)
        elif (face_M[1][2][0] == [-1, -1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtue4(1, 2)
            else:
                wtue42(1, 2)
        elif (face_M[1][2][0] == [-1, 1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtue32(1, 2)
            else:
                wtue3(1, 2)
        elif (face_M[1][2][0] == [0, -1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtu1(1, 2)
            else:
                wtu5(1, 2)
        elif (face_M[1][2][0] == [-1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtu2(1, 2)
            else:
                wtu6(1, 2)
        elif (face_M[1][2][0] == [0, 1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtu4(1, 2)
            else:
                wtu8(1, 2)
        elif (face_M[1][2][0] == [1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtu3(1, 2)
            else:
                wtu7(1, 2)


    #         white right*************************

    elif (face_M[0][5][1] == [0, 1, 0]):
        if (face_M[1][2][0] == [1, 1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtre1(1, 2)
            else:
                wtre12(1, 2)
        elif (face_M[1][2][0] == [1, -1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtre22(1, 2)
            else:
                wtre2(1, 2)
        elif (face_M[1][2][0] == [-1, -1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtre3(1, 2)
            else:
                wtre32(1, 2)
        elif (face_M[1][2][0] == [-1, 1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtre42(1, 2)
            else:
                wtre4(1, 2)
        elif (face_M[1][2][0] == [0, -1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtr2(1, 2)
            else:
                wtr6(1, 2)
        elif (face_M[1][2][0] == [-1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtr1(1, 2)
            else:
                wtr5(1, 2)
        elif (face_M[1][2][0] == [0, 1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtr4(1, 2)
            else:
                wtr8(1, 2)
        elif (face_M[1][2][0] == [1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtr3(1, 2)
            else:
                wtr7(1, 2)


    # white front***********************

    elif (face_M[0][5][1] == [1, 0, 0]):
        if (face_M[1][2][0] == [1, 1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtfe1(1, 2)
            else:
                wtfe12(1, 2)
        elif (face_M[1][2][0] == [1, -1, 0]):
            if (face_M[1][2][1] == [1, 0, 0]):
                wtfe22(1, 2)
            else:
                wtfe2(1, 2)
        elif (face_M[1][2][0] == [-1, -1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtfe3(1, 2)
            else:
                wtfe32(1, 2)
        elif (face_M[1][2][0] == [-1, 1, 0]):
            if (face_M[1][2][1] == [-1, 0, 0]):
                wtfe42(1, 2)
            else:
                wtfe4(1, 2)
        elif (face_M[1][2][0] == [0, -1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtf5(1, 2)
            else:
                wtf1(1, 2)
        elif (face_M[1][2][0] == [-1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtf6(1, 2)
            else:
                wtf2(1, 2)
        elif (face_M[1][2][0] == [0, 1, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtf7(1, 2)
            else:
                wtf3(1, 2)
        elif (face_M[1][2][0] == [1, 0, 1]):
            if (face_M[1][2][1] == [0, 0, 1]):
                wtf8(1, 2)
            else:
                wtf4(1, 2)
    print(
        "RED_GREEN DONE----------------------------------------------------------------------------------------------------------")
    printfaces()

    # **************RED_BLUE***top to position*******************************************************************************

    if (face_M[0][8][0] == [1, -1, -1] and face_M[0][8][1] == [0, 0, -1] and face_M[1][4][0] == [1, -1, 0] and
            face_M[1][4][1] == [1, 0, 0]):
        face_M[0][8][0] = [1, -1, -1]
    elif (face_M[0][8][0] == [1, 1, -1]):
        rotate(2, 1)
        rotate(5, 1)
        rotate(2, -1)
    elif (face_M[0][8][0] == [1, -1, -1]):
        rotate(3, -1)
        rotate(5, -1)
        rotate(3, 1)
        rotate(5, 1)
    elif (face_M[0][8][0] == [-1, -1, -1]):
        rotate(4, -1)
        rotate(5, -1)
        rotate(4, 1)
    elif (face_M[0][8][0] == [-1, 1, -1]):
        rotate(4, 1)
        rotate(5, 1)
        rotate(4, -1)
        rotate(5, 1)
    elif (face_M[0][8][0] == [1, 1, 1]):
        rotate(5, 1)
    elif (face_M[0][8][0] == [-1, -1, 1]):
        rotate(5, -1)
    elif (face_M[0][8][0] == [-1, 1, 1]):
        rotate(5, 1)
        rotate(5, 1)

    #     **************red_blue f2l***************

    # white up****************

    if (face_M[0][8][1] == [0, 0, 1]):
        if (face_M[3][2][0] == [1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtue1(1, 3)
            else:
                wtue12(1, 3)
        elif (face_M[3][2][0] == [-1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtue22(1, 3)
            else:
                wtue2(1, 3)
        elif (face_M[3][2][0] == [-1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtue4(1, 3)
            else:
                wtue42(1, 3)
        elif (face_M[3][2][0] == [1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtue32(1, 3)
            else:
                wtue3(1, 3)
        elif (face_M[3][2][0] == [-1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtu1(1, 3)
            else:
                wtu5(1, 3)
        elif (face_M[3][2][0] == [0, -1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtu3(1, 3)
            else:
                wtu7(1, 3)
        elif (face_M[3][2][0] == [1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtu4(1, 3)
            else:
                wtu8(1, 3)
        elif (face_M[3][2][0] == [0, 1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtu2(1, 3)
            else:
                wtu6(1, 3)


    #         white right*************************

    elif (face_M[0][8][1] == [1, 0, 0]):
        if (face_M[3][2][0] == [1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtre1(1, 3)
            else:
                wtre12(1, 3)
        elif (face_M[3][2][0] == [-1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtre22(1, 3)
            else:
                wtre2(1, 3)
        elif (face_M[3][2][0] == [-1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtre3(1, 3)
            else:
                wtre32(1, 3)
        elif (face_M[3][2][0] == [1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtre42(1, 3)
            else:
                wtre4(1, 3)
        elif (face_M[3][2][0] == [-1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtr2(1, 3)
            else:
                wtr6(1, 3)
        elif (face_M[3][2][0] == [0, 1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtr1(1, 3)
            else:
                wtr5(1, 3)
        elif (face_M[3][2][0] == [1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtr4(1, 3)
            else:
                wtr8(1, 3)
        elif (face_M[3][2][0] == [0, -1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtr3(1, 3)
            else:
                wtr7(1, 3)


    # white front***********************

    elif (face_M[0][8][1] == [0, -1, 0]):
        if (face_M[3][2][0] == [1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtfe1(1, 3)
            else:
                wtfe12(1, 3)
        elif (face_M[3][2][0] == [-1, -1, 0]):
            if (face_M[3][2][1] == [0, -1, 0]):
                wtfe22(1, 3)
            else:
                wtfe2(1, 3)
        elif (face_M[3][2][0] == [-1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtfe3(1, 3)
            else:
                wtfe32(1, 3)
        elif (face_M[3][2][0] == [1, 1, 0]):
            if (face_M[3][2][1] == [0, 1, 0]):
                wtfe42(1, 3)
            else:
                wtfe4(1, 3)
        elif (face_M[3][2][0] == [-1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtf5(1, 3)
            else:
                wtf1(1, 3)
        elif (face_M[3][2][0] == [0, 1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtf6(1, 3)
            else:
                wtf2(1, 3)
        elif (face_M[3][2][0] == [1, 0, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtf7(1, 3)
            else:
                wtf3(1, 3)
        elif (face_M[3][2][0] == [0, -1, 1]):
            if (face_M[3][2][1] == [0, 0, 1]):
                wtf8(1, 3)
            else:
                wtf4(1, 3)
    print(
        "RED_BLUE_DONE----------------------------------------------------------------------------------------------------------")
    printfaces()

    # **************ORANGE_BLUE***top to position*********************
    if (face_M[0][7][0] == [-1, -1, -1] and face_M[0][7][1] == [0, 0, -1] and face_M[4][2][0] == [-1, -1, 0] and
            face_M[4][2][1] == [-1, 0, 0]):
        face_M[0][7][0] = [-1, -1, -1]
    elif (face_M[0][7][0] == [-1, -1, -1]):
        rotate(4, -1)
        rotate(5, -1)
        rotate(4, 1)
        rotate(5, 1)
    elif (face_M[0][7][0] == [-1, 1, -1]):
        rotate(2, -1)
        rotate(5, -1)
        rotate(2, 1)
    elif (face_M[0][7][0] == [1, 1, -1]):
        rotate(1, -1)
        rotate(5, -1)
        rotate(1, 1)
        rotate(5, -1)
    elif (face_M[0][7][0] == [1, -1, -1]):
        rotate(1, 1)
        rotate(5, 1)
        rotate(1, -1)
    elif (face_M[0][7][0] == [-1, 1, 1]):
        rotate(5, -1)
    elif (face_M[0][7][0] == [1, 1, 1]):
        rotate(5, -1)
        rotate(5, -1)
    elif (face_M[0][7][0] == [1, -1, 1]):
        rotate(5, 1)

        #     **************orange_blue f2l********************************************************************************************

        # white up****************

    if (face_M[0][7][1] == [0, 0, 1]):
        if (face_M[4][2][0] == [-1, -1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtue1(4, 3)
            else:
                wtue12(4, 3)
        elif (face_M[4][2][0] == [-1, 1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtue22(4, 3)
            else:
                wtue2(4, 3)
        elif (face_M[4][2][0] == [1, 1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtue4(4, 3)
            else:
                wtue42(4, 3)
        elif (face_M[4][2][0] == [1, -1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtue32(4, 3)
            else:
                wtue3(4, 3)
        elif (face_M[4][2][0] == [0, 1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtu1(4, 3)
            else:
                wtu5(4, 3)
        elif (face_M[4][2][0] == [1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtu2(4, 3)
            else:
                wtu6(4, 3)
        elif (face_M[4][2][0] == [0, -1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtu4(4, 3)
            else:
                wtu8(4, 3)
        elif (face_M[4][2][0] == [-1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtu3(4, 3)
            else:
                wtu7(4, 3)

        #         white right*************************

    elif (face_M[0][7][1] == [0, -1, 0]):
        if (face_M[4][2][0] == [-1, -1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtre1(4, 3)
            else:
                wtre12(4, 3)
        elif (face_M[4][2][0] == [-1, 1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtre22(4, 3)
            else:
                wtre2(4, 3)
        elif (face_M[4][2][0] == [1, 1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtre3(4, 3)
            else:
                wtre32(4, 3)
        elif (face_M[4][2][0] == [1, -1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtre42(4, 3)
            else:
                wtre4(4, 3)
        elif (face_M[4][2][0] == [0, 1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtr2(4, 3)
            else:
                wtr6(4, 3)
        elif (face_M[4][2][0] == [1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtr1(4, 3)
            else:
                wtr5(4, 3)
        elif (face_M[4][2][0] == [0, -1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtr4(4, 3)
            else:
                wtr8(4, 3)
        elif (face_M[4][2][0] == [-1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtr3(4, 3)
            else:
                wtr7(4, 3)

        # white front***********************

    elif (face_M[0][7][1] == [-1, 0, 0]):
        if (face_M[4][2][0] == [-1, -1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtfe1(4, 3)
            else:
                wtfe12(4, 3)
        elif (face_M[4][2][0] == [-1, 1, 0]):
            if (face_M[4][2][1] == [-1, 0, 0]):
                wtfe22(4, 3)
            else:
                wtfe2(4, 3)
        elif (face_M[4][2][0] == [1, 1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtfe3(4, 3)
            else:
                wtfe32(4, 3)
        elif (face_M[4][2][0] == [1, -1, 0]):
            if (face_M[4][2][1] == [1, 0, 0]):
                wtfe42(4, 3)
            else:
                wtfe4(4, 3)
        elif (face_M[4][2][0] == [0, 1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtf5(4, 3)
            else:
                wtf1(4, 3)
        elif (face_M[4][2][0] == [1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtf6(4, 3)
            else:
                wtf2(4, 3)
        elif (face_M[4][2][0] == [0, -1, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtf7(4, 3)
            else:
                wtf3(4, 3)
        elif (face_M[4][2][0] == [-1, 0, 1]):
            if (face_M[4][2][1] == [0, 0, 1]):
                wtf8(4, 3)
            else:
                wtf4(4, 3)
    print(
        "ORANGE_BLUE_DONE----------------------------------------------------------------------------------------------------------")
    printfaces()

    # **************ORANGE_GREEN***top to position*******************************************************************************
    if (face_M[0][6][0] == [-1, 1, -1] and face_M[0][6][1] == [0, 0, -1] and face_M[4][4][0] == [-1, 1, 0] and
            face_M[4][4][1] == [-1, 0, 0]):
        face_M[0][6][0] = [-1, 1, -1]
    elif (face_M[0][6][0] == [1, 1, -1]):
        rotate(1, -1)
        rotate(5, -1)
        rotate(1, 1)
    elif (face_M[0][6][0] == [1, -1, -1]):
        rotate(3, -1)
        rotate(5, -1)
        rotate(3, 1)
        rotate(5, -1)
    elif (face_M[0][6][0] == [-1, -1, -1]):
        rotate(3, 1)
        rotate(5, 1)
        rotate(3, -1)
    elif (face_M[0][6][0] == [-1, 1, -1]):
        rotate(4, 1)
        rotate(5, 1)
        rotate(4, -1)
        rotate(5, -1)
    elif (face_M[0][6][0] == [1, 1, 1]):
        rotate(5, -1)
    elif (face_M[0][6][0] == [-1, -1, 1]):
        rotate(5, 1)
    elif (face_M[0][6][0] == [1, -1, 1]):
        rotate(5, 1)
        rotate(5, 1)

        #     **************orange_GREEN f2l***************

        # white up****************

    if (face_M[0][6][1] == [0, 0, 1]):
        if (face_M[2][2][0] == [-1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtue1(4, 2)
            else:
                wtue12(4, 2)
        elif (face_M[2][2][0] == [1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtue22(4, 2)
            else:
                wtue2(4, 2)
        elif (face_M[2][2][0] == [1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtue4(4, 2)
            else:
                wtue42(4, 2)
        elif (face_M[2][2][0] == [-1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtue32(4, 2)
            else:
                wtue3(4, 2)
        elif (face_M[2][2][0] == [1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtu1(4, 2)
            else:
                wtu5(4, 2)
        elif (face_M[2][2][0] == [0, 1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtu3(4, 2)
            else:
                wtu7(4, 2)
        elif (face_M[2][2][0] == [-1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtu4(4, 2)
            else:
                wtu8(4, 2)
        elif (face_M[2][2][0] == [0, -1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtu2(4, 2)
            else:
                wtu6(4, 2)

        #         white right*************************

    elif (face_M[0][6][1] == [-1, 0, 0]):
        if (face_M[2][2][0] == [-1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtre1(4, 2)
            else:
                wtre12(4, 2)
        elif (face_M[2][2][0] == [1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtre22(4, 2)
            else:
                wtre2(4, 2)
        elif (face_M[2][2][0] == [1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtre3(4, 2)
            else:
                wtre32(4, 2)
        elif (face_M[2][2][0] == [-1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtre42(4, 2)
            else:
                wtre4(4, 2)
        elif (face_M[2][2][0] == [1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtr2(4, 2)
            else:
                wtr6(4, 2)
        elif (face_M[2][2][0] == [0, -1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtr1(4, 2)
            else:
                wtr5(4, 2)
        elif (face_M[2][2][0] == [-1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtr4(4, 2)
            else:
                wtr8(4, 2)
        elif (face_M[2][2][0] == [0, 1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtr3(4, 2)
            else:
                wtr7(4, 2)

        # white front***********************

    elif (face_M[0][6][1] == [0, 1, 0]):
        if (face_M[2][2][0] == [-1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtfe1(4, 2)
            else:
                wtfe12(4, 2)
        elif (face_M[2][2][0] == [1, 1, 0]):
            if (face_M[2][2][1] == [0, 1, 0]):
                wtfe22(4, 2)
            else:
                wtfe2(4, 2)
        elif (face_M[2][2][0] == [1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtfe3(4, 2)
            else:
                wtfe32(4, 2)
        elif (face_M[2][2][0] == [-1, -1, 0]):
            if (face_M[2][2][1] == [0, -1, 0]):
                wtfe42(4, 2)
            else:
                wtfe4(4, 2)
        elif (face_M[2][2][0] == [1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtf5(4, 2)
            else:
                wtf1(4, 2)
        elif (face_M[2][2][0] == [0, -1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtf6(4, 2)
            else:
                wtf2(4, 2)
        elif (face_M[2][2][0] == [-1, 0, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtf7(4, 2)
            else:
                wtf3(4, 2)
        elif (face_M[2][2][0] == [0, 1, 1]):
            if (face_M[2][2][1] == [0, 0, 1]):
                wtf8(4, 2)
            else:
                wtf4(4, 2)


def bplus():
    # ****************BLUE*******************************************
    if face_M[0][4][0] == [0, -1, -1]:
        if face_M[0][4][1] == [0, -1, 0]:
            rotate(3, 1)

            rotate(4, 1)

            rotate(0, 1)

    elif face_M[0][4][0] == [-1, 0, -1]:
        if face_M[0][4][1] == [0, 0, -1]:
            rotate(0, 1)

        elif face_M[0][4][1] == [-1, 0, 0]:
            rotate(4, -1)

            rotate(3, -1)

    elif face_M[0][4][0] == [0, 1, -1]:
        if face_M[0][4][1] == [0, 0, -1]:
            rotate(0, 1)
            rotate(0, 1)
        elif face_M[0][4][1] == [0, 1, 0]:
            rotate(2, 1)
            rotate(1, 1)
            rotate(0, -1)

    elif face_M[0][4][0] == [1, 0, -1]:
        if face_M[0][4][1] == [0, 0, -1]:
            rotate(0, -1)
        elif face_M[0][4][1] == [1, 0, 0]:
            rotate(1, 1)
            rotate(3, 1)
    elif face_M[0][4][0] == [-1, -1, 0]:
        if face_M[0][4][1] == [-1, 0, 0]:
            rotate(3, -1)
        elif face_M[0][4][1] == [0, -1, 0]:
            rotate(4, 1)
            rotate(0, 1)
    elif face_M[0][4][0] == [-1, 1, 0]:
        if face_M[0][4][1] == [-1, 0, 0]:
            rotate(4, -1)
            rotate(4, -1)
            rotate(3, -1)
        elif face_M[0][4][1] == [0, 1, 0]:
            rotate(4, -1)
            rotate(0, 1)
    elif face_M[0][4][0] == [1, 1, 0]:
        if face_M[0][4][1] == [1, 0, 0]:
            rotate(2, -1)
            rotate(0, -1)
            rotate(0, -1)
        elif face_M[0][4][1] == [0, 1, 0]:
            rotate(1, 1)
            rotate(0, -1)
    elif face_M[0][4][0] == [1, -1, 0]:
        if face_M[0][4][1] == [1, 0, 0]:
            rotate(3, 1)

        elif face_M[0][4][1] == [0, -1, 0]:
            rotate(1, -1)
            rotate(0, -1)
    elif face_M[0][4][0] == [0, -1, 1]:
        if face_M[0][4][1] == [0, 0, 1]:
            rotate(3, 1)
            rotate(3, 1)

        elif face_M[0][4][1] == [0, -1, 0]:
            rotate(3, -1)
            rotate(4, 1)
            rotate(0, 1)
    elif face_M[0][4][0] == [-1, 0, 1]:
        if face_M[0][4][1] == [0, 0, 1]:
            rotate(5, -1)
            rotate(3, 1)
            rotate(3, 1)

        elif face_M[0][4][1] == [-1, 0, 0]:
            rotate(4, 1)
            rotate(3, -1)
    elif face_M[0][4][0] == [0, 1, 1]:
        if face_M[0][4][1] == [0, 0, 1]:
            rotate(5, -1)
            rotate(5, -1)

            rotate(3, 1)
            rotate(3, 1)

        elif face_M[0][4][1] == [0, 1, 0]:
            rotate(5, -1)
            rotate(4, 1)
            rotate(3, -1)
    elif face_M[0][4][0] == [1, 0, 1]:
        if face_M[0][4][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(3, 1)
            rotate(3, 1)

        elif face_M[0][4][1] == [1, 0, 0]:
            rotate(1, -1)
            rotate(3, 1)

    # ********************************ORANGE*********************************
    if face_M[0][3][0] == [-1, 0, -1]:

        if face_M[0][3][1] == [-1, 0, 0]:
            rotate(4, -1)
            rotate(3, -1)
            rotate(0, -1)
            rotate(3, 1)
    elif face_M[0][3][0] == [0, 1, -1]:
        if face_M[0][3][1] == [0, 0, -1]:
            rotate(2, -1)
            rotate(0, -1)
            rotate(2, 1)
            rotate(0, 1)
        elif face_M[0][3][1] == [0, 1, 0]:
            rotate(2, -1)
            rotate(4, -1)


    elif face_M[0][3][0] == [1, 0, -1]:
        if face_M[0][3][1] == [0, 0, -1]:
            rotate(1, -1)
            rotate(2, -1)
            rotate(2, -1)
            rotate(4, -1)

        elif face_M[0][3][1] == [1, 0, 0]:
            rotate(1, -1)
            rotate(0, -1)
            rotate(2, -1)
            rotate(0, 1)
    elif face_M[0][3][0] == [-1, -1, 0]:
        if face_M[0][3][1] == [-1, 0, 0]:
            rotate(0, 1)
            rotate(3, -1)
            rotate(0, -1)

        elif face_M[0][3][1] == [0, -1, 0]:
            rotate(4, 1)

    elif face_M[0][3][0] == [-1, 1, 0]:
        if face_M[0][3][1] == [-1, 0, 0]:
            rotate(0, -1)
            rotate(2, 1)
            rotate(0, 1)
        elif face_M[0][3][1] == [0, 1, 0]:
            rotate(4, -1)
    elif face_M[0][3][0] == [1, 1, 0]:
        if face_M[0][3][1] == [1, 0, 0]:
            rotate(0, -1)
            rotate(2, -1)
            rotate(0, 1)
        elif face_M[0][3][1] == [0, 1, 0]:
            rotate(2, -1)
            rotate(2, -1)
            rotate(4, -1)
    elif face_M[0][3][0] == [1, -1, 0]:
        if face_M[0][3][1] == [1, 0, 0]:
            rotate(0, 1)
            rotate(3, 1)
            rotate(0, -1)

        elif face_M[0][3][1] == [0, -1, 0]:
            rotate(3, -1)
            rotate(3, -1)
            rotate(4, 1)
            rotate(3, -1)
            rotate(3, -1)
    elif face_M[0][3][0] == [0, -1, 1]:
        if face_M[0][3][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(4, 1)
            rotate(4, 1)

        elif face_M[0][3][1] == [0, -1, 0]:
            rotate(3, -1)
            rotate(4, 1)
            rotate(3, 1)
    elif face_M[0][3][0] == [-1, 0, 1]:
        if face_M[0][3][1] == [0, 0, 1]:
            rotate(4, 1)
            rotate(4, 1)

        elif face_M[0][3][1] == [-1, 0, 0]:
            rotate(5, -1)
            rotate(3, -1)
            rotate(4, 1)
            rotate(3, 1)
    elif face_M[0][3][0] == [0, 1, 1]:
        if face_M[0][3][1] == [0, 0, 1]:
            rotate(5, -1)
            rotate(4, 1)
            rotate(4, 1)

        elif face_M[0][3][1] == [0, 1, 0]:
            rotate(2, 1)
            rotate(4, -1)
    elif face_M[0][3][0] == [1, 0, 1]:
        if face_M[0][3][1] == [0, 0, 1]:
            rotate(5, -1)
            rotate(5, -1)
            rotate(4, 1)
            rotate(4, 1)

        elif face_M[0][3][1] == [1, 0, 0]:
            rotate(5, -1)
            rotate(2, 1)
            rotate(4, -1)

    # *********************GREEN**********************************
    if face_M[0][2][0] == [0, 1, -1]:
        if face_M[0][2][1] == [0, 1, 0]:
            rotate(2, 1)

            rotate(0, -1)

            rotate(1, 1)

            rotate(0, 1)



    elif face_M[0][2][0] == [1, 0, -1]:
        if face_M[0][2][1] == [0, 0, -1]:
            rotate(1, -1)
            rotate(0, -1)
            rotate(1, 1)
            rotate(0, 1)

        elif face_M[0][2][1] == [1, 0, 0]:
            rotate(1, -1)
            rotate(2, -1)

    elif face_M[0][2][0] == [-1, -1, 0]:
        if face_M[0][2][1] == [-1, 0, 0]:
            rotate(0, 1)
            rotate(0, 1)
            rotate(3, -1)
            rotate(0, -1)
            rotate(0, -1)

        elif face_M[0][2][1] == [0, -1, 0]:
            rotate(0, 1)
            rotate(4, 1)
            rotate(0, -1)

    elif face_M[0][2][0] == [-1, 1, 0]:
        if face_M[0][2][1] == [-1, 0, 0]:
            rotate(2, 1)

        elif face_M[0][2][1] == [0, 1, 0]:
            rotate(0, 1)
            rotate(4, -1)
            rotate(0, -1)
    elif face_M[0][2][0] == [1, 1, 0]:
        if face_M[0][2][1] == [1, 0, 0]:

            rotate(2, -1)

        elif face_M[0][2][1] == [0, 1, 0]:
            rotate(0, -1)
            rotate(1, 1)
            rotate(0, 1)
    elif face_M[0][2][0] == [1, -1, 0]:
        if face_M[0][2][1] == [1, 0, 0]:
            rotate(0, 1)
            rotate(0, 1)

            rotate(3, 1)
            rotate(0, -1)
            rotate(0, -1)

        elif face_M[0][2][1] == [0, -1, 0]:
            rotate(0, -1)
            rotate(1, -1)
            rotate(0, 1)
    elif face_M[0][2][0] == [0, -1, 1]:
        if face_M[0][2][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(5, 1)
            rotate(2, 1)
            rotate(2, 1)

        elif face_M[0][2][1] == [0, -1, 0]:
            rotate(5, -1)
            rotate(1, 1)
            rotate(2, -1)
    elif face_M[0][2][0] == [-1, 0, 1]:
        if face_M[0][2][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(2, 1)
            rotate(2, 1)

        elif face_M[0][2][1] == [-1, 0, 0]:
            rotate(4, -1)
            rotate(2, 1)
            rotate(4, 1)
    elif face_M[0][2][0] == [0, 1, 1]:
        if face_M[0][2][1] == [0, 0, 1]:

            rotate(2, 1)
            rotate(2, 1)

        elif face_M[0][2][1] == [0, 1, 0]:
            rotate(5, 1)
            rotate(1, 1)
            rotate(2, -1)
    elif face_M[0][2][0] == [1, 0, 1]:
        if face_M[0][2][1] == [0, 0, 1]:
            rotate(5, -1)
            rotate(2, 1)
            rotate(2, 1)

        elif face_M[0][2][1] == [1, 0, 0]:
            rotate(1, 1)
            rotate(2, -1)

    # *************************************RED*******************************

    if face_M[0][1][0] == [1, 0, -1]:
        if face_M[0][1][1] == [1, 0, 0]:
            rotate(1, -1)
            rotate(0, 1)
            rotate(2, -1)
            rotate(0, -1)

    elif face_M[0][1][0] == [-1, -1, 0]:
        if face_M[0][1][1] == [-1, 0, 0]:
            rotate(0, -1)
            rotate(3, -1)
            rotate(0, 1)


        elif face_M[0][1][1] == [0, -1, 0]:
            rotate(0, 1)
            rotate(0, 1)
            rotate(4, 1)
            rotate(0, -1)
            rotate(0, -1)

    elif face_M[0][1][0] == [-1, 1, 0]:
        if face_M[0][1][1] == [-1, 0, 0]:
            rotate(0, 1)
            rotate(2, 1)
            rotate(0, -1)

        elif face_M[0][1][1] == [0, 1, 0]:
            rotate(0, 1)
            rotate(0, 1)
            rotate(4, -1)
            rotate(0, -1)
            rotate(0, -1)
    elif face_M[0][1][0] == [1, 1, 0]:
        if face_M[0][1][1] == [1, 0, 0]:
            rotate(0, 1)

            rotate(2, -1)
            rotate(0, -1)

        elif face_M[0][1][1] == [0, 1, 0]:

            rotate(1, 1)

    elif face_M[0][1][0] == [1, -1, 0]:
        if face_M[0][1][1] == [1, 0, 0]:
            rotate(0, -1)

            rotate(3, 1)
            rotate(0, 1)


        elif face_M[0][1][1] == [0, -1, 0]:

            rotate(1, -1)

    elif face_M[0][1][0] == [0, -1, 1]:
        if face_M[0][1][1] == [0, 0, 1]:
            rotate(5, -1)

            rotate(1, 1)
            rotate(1, 1)

        elif face_M[0][1][1] == [0, -1, 0]:
            rotate(3, 1)
            rotate(1, -1)
            rotate(3, -1)
    elif face_M[0][1][0] == [-1, 0, 1]:
        if face_M[0][1][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(5, 1)
            rotate(1, 1)
            rotate(1, 1)

        elif face_M[0][1][1] == [-1, 0, 0]:
            rotate(5, -1)
            rotate(3, 1)
            rotate(1, -1)
            rotate(3, -1)
    elif face_M[0][1][0] == [0, 1, 1]:
        if face_M[0][1][1] == [0, 0, 1]:
            rotate(5, 1)
            rotate(1, 1)
            rotate(1, 1)

        elif face_M[0][1][1] == [0, 1, 0]:
            rotate(2, -1)
            rotate(1, 1)
            rotate(2, 1)
    elif face_M[0][1][0] == [1, 0, 1]:
        if face_M[0][1][1] == [0, 0, 1]:
            rotate(1, 1)
            rotate(1, 1)

        elif face_M[0][1][1] == [1, 0, 0]:
            rotate(5, 1)
            rotate(3, 1)
            rotate(1, -1)
            rotate(3, -1)
    print('steps are', steps)
    print('total steps after optimization are ', cntr)


def topcross(topface):
    while (topface[0][1] != 5 or topface[2][1] != 5 or topface[1][0] != 5 or topface[1][2] != 5):
        u, r, l, d, f = urldf(1, 2)
        if (topface[0][1] == 5 and topface[2][1] == 5 and topface[1][0] == 5 and topface[1][2] == 5):
            break
        elif (topface[0][1] == 5 and topface[1][2] == 5):
            u, r, l, d, f = urldf(1, 3)

            rotate(f, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(f, -1)
        elif (topface[2][1] == 5 and topface[1][2] == 5):
            u, r, l, d, f = urldf(4, 3)
            rotate(f, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(f, -1)
        elif (topface[2][1] == 5 and topface[1][0] == 5):
            u, r, l, d, f = urldf(2, 4)
            rotate(f, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(f, -1)
        elif (topface[0][1] == 5 and topface[1][0] == 5):
            u, r, l, d, f = urldf(1, 2)
            rotate(f, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(f, -1)
        elif (topface[0][1] == 5 and topface[2][1] == 5):
            u, r, l, d, f = urldf(1, 3)
            rotate(f, 1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(f, -1)
        elif (topface[1][0] == 5 and topface[1][2] == 5):
            u, r, l, d, f = urldf(1, 2)
            rotate(f, 1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(f, -1)
        else:
            rotate(f, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(f, -1)
        print(
            "----------------------------------------------------------------------------------------------------------")
        allface = printfaces()
        topface = allface[5]
    print("total steps are ", steps)
    print('total steps after optimization are ', cntr)


def oll(allface):
    topf = allface[5]
    # OLL CASE-1==4
    if (topf[0][0] == 5 and allface[4][0][0] == 5 and allface[2][0][0] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(2, 4)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    elif (topf[0][2] == 5 and allface[3][0][0] == 5 and allface[2][0][0] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(2, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    elif (topf[2][2] == 5 and allface[3][0][0] == 5 and allface[4][0][0] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(3, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    elif (topf[2][0] == 5 and allface[3][0][0] == 5 and allface[4][0][0] == 5 and allface[2][0][0] == 5):
        u, r, l, d, f = urldf(4, 3)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    # OLL CASE-2==4
    elif (topf[0][0] == 5 and allface[2][0][2] == 5 and allface[1][0][2] == 5 and allface[3][0][2] == 5):
        u, r, l, d, f = urldf(3, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (topf[0][2] == 5 and allface[4][0][2] == 5 and allface[1][0][2] == 5 and allface[3][0][2] == 5):
        u, r, l, d, f = urldf(3, 4)

        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (topf[2][2] == 5 and allface[2][0][2] == 5 and allface[4][0][2] == 5 and allface[3][0][2] == 5):
        u, r, l, d, f = urldf(2, 4)
        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (topf[2][0] == 5 and allface[2][0][2] == 5 and allface[1][0][2] == 5 and allface[4][0][2] == 5):

        u, r, l, d, f = urldf(2, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    #         OLL3==2
    elif (allface[4][0][0] == 5 and allface[4][0][2] == 5 and allface[1][0][2] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(2, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    elif (allface[2][0][0] == 5 and allface[2][0][2] == 5 and allface[3][0][2] == 5 and allface[3][0][0] == 5):
        u, r, l, d, f = urldf(3, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
    #     OLL case4==4
    elif (allface[4][0][0] == 5 and allface[1][0][2] == 5 and allface[3][0][2] == 5 and allface[3][0][0] == 5):
        u, r, l, d, f = urldf(2, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(u, -1)
        rotate(r, 1)
    elif (allface[2][0][0] == 5 and allface[3][0][2] == 5 and allface[4][0][2] == 5 and allface[4][0][0] == 5):
        u, r, l, d, f = urldf(3, 1)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(u, -1)
        rotate(r, 1)
    elif (allface[1][0][0] == 5 and allface[4][0][2] == 5 and allface[2][0][2] == 5 and allface[2][0][0] == 5):
        u, r, l, d, f = urldf(3, 4)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(u, -1)
        rotate(r, 1)
    elif (allface[3][0][0] == 5 and allface[2][0][2] == 5 and allface[1][0][2] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(2, 4)
        rotate(r, 1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(r, 1)
        rotate(r, 1)
        rotate(u, -1)
        rotate(r, -1)
        rotate(r, -1)
        rotate(u, -1)
        rotate(u, -1)
        rotate(r, 1)
    #     OLL CASE5==4
    elif (allface[5][0][2] == 5 and allface[5][2][2] == 5 and allface[4][0][2] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(2, 1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
        rotate(f, -1)
    elif (allface[5][2][0] == 5 and allface[5][2][2] == 5 and allface[2][0][2] == 5 and allface[3][0][0] == 5):
        u, r, l, d, f = urldf(3, 1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
        rotate(f, -1)
    elif (allface[5][2][0] == 5 and allface[5][0][0] == 5 and allface[1][0][2] == 5 and allface[4][0][0] == 5):
        u, r, l, d, f = urldf(4, 3)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
        rotate(f, -1)
    elif (allface[5][0][0] == 5 and allface[5][0][2] == 5 and allface[3][0][2] == 5 and allface[2][0][0] == 5):
        u, r, l, d, f = urldf(4, 2)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
        rotate(f, -1)
    #     OLL CASE6==4
    elif (allface[5][0][0] == 5 and allface[5][2][2] == 5 and allface[1][0][0] == 5 and allface[2][0][2] == 5):
        u, r, l, d, f = urldf(4, 2)
        rotate(f, -1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
    elif (allface[5][0][2] == 5 and allface[5][2][0] == 5 and allface[1][0][2] == 5 and allface[3][0][0] == 5):
        u, r, l, d, f = urldf(1, 2)
        rotate(f, -1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
    elif (allface[5][0][0] == 5 and allface[5][2][2] == 5 and allface[3][0][2] == 5 and allface[4][0][0] == 5):
        u, r, l, d, f = urldf(1, 3)
        rotate(f, -1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
    elif (allface[5][0][2] == 5 and allface[5][2][0] == 5 and allface[4][0][2] == 5 and allface[2][0][0] == 5):
        u, r, l, d, f = urldf(4, 3)
        rotate(f, -1)
        rotate(l, 1)
        rotate(f, 1)
        rotate(r, -1)
        rotate(f, -1)
        rotate(l, -1)
        rotate(f, 1)
        rotate(r, 1)
    #     OLL CASES7==4
    elif (allface[5][0][2] == 5 and allface[5][0][0] == 5 and allface[1][0][2] == 5 and allface[1][0][0] == 5):
        u, r, l, d, f = urldf(1, 2)
        rotate(r, 1)
        rotate(r, 1)
        rotate(d, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(d, -1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (allface[5][0][2] == 5 and allface[5][2][2] == 5 and allface[3][0][2] == 5 and allface[3][0][0] == 5):
        u, r, l, d, f = urldf(1, 3)
        rotate(r, 1)
        rotate(r, 1)
        rotate(d, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(d, -1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (allface[5][2][2] == 5 and allface[5][2][0] == 5 and allface[4][0][2] == 5 and allface[4][0][0] == 5):
        u, r, l, d, f = urldf(4, 3)
        rotate(r, 1)
        rotate(r, 1)
        rotate(d, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(d, -1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    elif (allface[5][0][0] == 5 and allface[5][2][0] == 5 and allface[2][0][2] == 5 and allface[2][0][0] == 5):
        u, r, l, d, f = urldf(4, 2)
        rotate(r, 1)
        rotate(r, 1)
        rotate(d, 1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, 1)
        rotate(d, -1)
        rotate(r, -1)
        rotate(u, 1)
        rotate(u, 1)
        rotate(r, -1)
    print("steps tILL OLL= ", steps)
    print('total steps after optimization are ', cntr)

def topcorners(allface):
    while(allface[1][0][0]!=allface[1][0][2] or allface[2][0][0]!=allface[2][0][2] or allface[3][0][0]!=allface[3][0][2] or allface[4][0][0]!=allface[4][0][2]):
        if (allface[1][0][0]==allface[1][0][2]):
            u,r,l,d,f=urldf(4,3)
            b=back(f)
            rotate(r,-1)
            rotate(f,1)
            rotate(r,-1)
            rotate(b,-1)
            rotate(b,-1)
            rotate(r, 1)
            rotate(f, -1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r,-1)
            rotate(r,-1)
        elif (allface[2][0][0]==allface[2][0][2]):
            u,r,l,d,f=urldf(1,3)
            b=back(f)
            rotate(r,-1)
            rotate(f,1)
            rotate(r,-1)
            rotate(b,-1)
            rotate(b,-1)
            rotate(r, 1)
            rotate(f, -1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r,-1)
            rotate(r,-1)
        elif (allface[4][0][0]==allface[4][0][2]):
            u,r,l,d,f=urldf(1,2)
            b=back(f)
            rotate(r,-1)
            rotate(f,1)
            rotate(r,-1)
            rotate(b,-1)
            rotate(b,-1)
            rotate(r, 1)
            rotate(f, -1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r,-1)
            rotate(r,-1)
        elif (allface[3][0][0]==allface[3][0][2]):
            u,r,l,d,f=urldf(4,2)
            b=back(f)
            rotate(r,-1)
            rotate(f,1)
            rotate(r,-1)
            rotate(b,-1)
            rotate(b,-1)
            rotate(r, 1)
            rotate(f, -1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r,-1)
            rotate(r,-1)
        else:
            u, r, l, d, f = urldf(4, 3)
            b = back(f)
            rotate(r, -1)
            rotate(f, 1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r, 1)
            rotate(f, -1)
            rotate(r, -1)
            rotate(b, -1)
            rotate(b, -1)
            rotate(r, -1)
            rotate(r, -1)
        allface=printfaces()
    if (allface[1][0][0]==allface[1][0][2] and allface[2][0][0]==allface[2][0][2] and allface[3][0][0]==allface[3][0][2] and allface[4][0][0]==allface[4][0][2]):
        if allface[1][0][0]==3:
            rotate(5,1)
        elif allface[1][0][0]==4:
            rotate(5,1)
            rotate(5,1)
        elif allface[1][0][0]==2:
            rotate(5,-1)
    print("steps till top corner are ",steps)
    print('total steps after optimization are ', cntr)

def topedge(allface):
    while(allface[1][0][1]!=1 or allface[2][0][1]!=2 or allface[3][0][1]!=3 or allface[4][0][1]!=4):
        if allface[1][0][1]==1 and allface[2][0][1]==3:
            u,r,l,d,f=urldf(3,4)
            rotate(r,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, 1)
            rotate(r, -1)
        elif allface[1][0][1]==1 and allface[2][0][1]==4:
            u,r,l,d,f=urldf(3,4)
            rotate(r,1)
            rotate(u,-1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, 1)
            rotate(r, 1)
        elif allface[3][0][1]==3 and allface[2][0][1]==1:
            u,r,l,d,f=urldf(2,4)
            rotate(r,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, 1)
            rotate(r, -1)
        elif allface[3][0][1]==3 and allface[2][0][1]==4:
            u,r,l,d,f=urldf(2,4)
            rotate(r,1)
            rotate(u,-1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, 1)
            rotate(r, 1)
        elif allface[4][0][1]==4 and allface[2][0][1]==1:
            u,r,l,d,f=urldf(2,1)
            rotate(r,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, 1)
            rotate(r, -1)
        elif allface[4][0][1]==4 and allface[2][0][1]==3:
            u,r,l,d,f=urldf(2,1)
            rotate(r,1)
            rotate(u,-1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, 1)
            rotate(r, 1)
        elif allface[2][0][1]==2 and allface[1][0][1]==3:
            u,r,l,d,f=urldf(1,3)
            rotate(r,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, 1)
            rotate(r, -1)
        elif allface[2][0][1]==2 and allface[1][0][1]==4:
            u,r,l,d,f=urldf(1,3)
            rotate(r,1)
            rotate(u,-1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, 1)
            rotate(r, 1)
        elif allface[3][0][1]==2 and allface[2][0][1]==3 and allface[1][0][1]==4 and allface[4][0][1]==1:
            u,r,l,d,f=urldf(2,4)
            rotate(r,1)
            rotate(r,1)
            rotate(l,1)
            rotate(l,1)
            rotate(d,1)
            rotate(r, 1)
            rotate(r, 1)
            rotate(l, 1)
            rotate(l, 1)
            rotate(u, 1)
            rotate(u, 1)
            rotate(r, 1)
            rotate(r, 1)
            rotate(l, 1)
            rotate(l, 1)
            rotate(d, 1)
            rotate(r, 1)
            rotate(r, 1)
            rotate(l, 1)
            rotate(l, 1)
        else:
            u,r,l,d,f=urldf(3,4)
            rotate(r,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r,1)
            rotate(u,1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, -1)
            rotate(r, -1)
            rotate(u, 1)
            rotate(r, -1)
        allface=printfaces()
    print("steps till top edges are ", steps)
    print('total steps after optimization are ', cntr)




img = np.empty(shape=(6), dtype='object')

cv2.waitKey(0)

if __name__ == "__main__":
    block = np.empty(shape=(3, 3, 3), dtype='object')
    face_M = np.empty(shape=(6, 9, 2), dtype='object')
    block_part = np.empty(shape=(3, 3), dtype='object')
    img[0] = cv2.imread('red.jpg')
    img[1] = cv2.imread('blue.jpg')
    img[2] = cv2.imread('orange.jpg')
    img[3] = cv2.imread('green.jpg')
    img[4] = cv2.imread('white.jpg')
    img[5] = cv2.imread('yellow.jpg')
    t = 0

    # **************input**********************************
    while (t < 6):
        x = img[t].shape[0] / 3
        y = img[t].shape[1] / 3

        hsv = cv2.cvtColor(img[t], cv2.COLOR_BGR2HSV)
        cv2.line(hsv, (int(hsv.shape[1] / 3), 0), (int(hsv.shape[1] / 3), hsv.shape[0]), (0, 0, 0), 10, 1)
        cv2.line(hsv, (0, int(hsv.shape[0] / 3)), (hsv.shape[1], int(hsv.shape[0] / 3)), (0, 0, 0), 10, 1)
        cv2.line(hsv, (int(hsv.shape[1] * 2 / 3), 0), (int(hsv.shape[1] * 2 / 3), hsv.shape[0]), (0, 0, 0), 10, 1)
        cv2.line(hsv, (0, int(hsv.shape[0] * 2 / 3)), (hsv.shape[1], int(hsv.shape[0] * 2 / 3)), (0, 0, 0), 10, 1)
        cv2.imshow('hsv', hsv)
        # ---------FOR_RED--------------
        lv_red = np.array([0, 10, 100])
        uv_red = np.array([5, 255, 255])
        lv2_red = np.array([156, 0, 32])
        uv2_red = np.array([255, 255, 255])
        mask_red = cv2.inRange(hsv, lv_red, uv_red)
        mask2_red = cv2.inRange(hsv, lv2_red, uv2_red)
        res_red = cv2.addWeighted(mask_red, 1, mask2_red, 1, 0)
        res1_red = cv2.bitwise_and(img[t], img[t], mask=res_red)
        res1_red[np.where((res1_red == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_r, _ = cv2.findContours(res_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for cnt in ct_r:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 1
                cv2.drawContours(img[t], cnt, -1, (0, 255, 0), 3)
                cv2.putText(img[t], 'R', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # ----------FOR__Yellow------------
        lv_yellow = np.array([19, 213, 0])
        uv_yellow = np.array([39, 255, 255])
        mask_yellow = cv2.inRange(hsv, lv_yellow, uv_yellow)
        res_yellow = cv2.bitwise_and(img[t], img[t], mask=mask_yellow)
        res_yellow[np.where((res_yellow == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_y, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in ct_y:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 5
                cv2.drawContours(img[t], cnt, -1, (0, 255, 0), 3)
                cv2.putText(img[t], 'y', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # ----------FOR__orange------------
        lv_orange = np.array([8, 170, 162])
        uv_orange = np.array([18, 255, 255])
        mask_orange = cv2.inRange(hsv, lv_orange, uv_orange)
        res_orange = cv2.bitwise_and(img[t], img[t], mask=mask_orange)
        res_orange[np.where((res_orange == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_o, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in ct_o:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 4
                cv2.drawContours(img[t], cnt, -1, (0, 255, 0), 3)
                cv2.putText(img[t], 'o', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # ----------FOR__blue------------
        lv_blue = np.array([75, 145, 131])
        uv_blue = np.array([119, 255, 255])
        mask_blue = cv2.inRange(hsv, lv_blue, uv_blue)
        res_blue = cv2.bitwise_and(img[t], img[t], mask=mask_blue)
        res_blue[np.where((res_blue == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_b, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in ct_b:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 3
                cv2.drawContours(img[t], cnt, -1, (0, 255, 0), 3)
                cv2.putText(img[t], 'b', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # ----------FOR__GREEN------------
        lv_green = np.array([48, 145, 131])
        uv_green = np.array([63, 255, 255])
        mask_green = cv2.inRange(hsv, lv_green, uv_green)
        res_green = cv2.bitwise_and(img[t], img[t], mask=mask_green)
        res_green[np.where((res_green == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_g, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in ct_g:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 2
                cv2.drawContours(img[t], cnt, -1, (0, 255, 0), 3)
                cv2.putText(img[t], 'g', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

        # ----------FOR__WHITE------------
        lv_white = np.array([25, 0, 0])
        uv_white = np.array([180, 89, 255])
        mask_white = cv2.inRange(hsv, lv_white, uv_white)
        res_white = cv2.bitwise_and(img[t], img[t], mask=mask_white)
        res_white[np.where((res_white == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        ct_w, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in ct_w:
            area = cv2.contourArea(cnt)
            cen = cv2.moments(cnt)
            if area > 5000:
                approx = cv2.approxPolyDP(cnt, 0.5 * cv2.arcLength(cnt, True), True)
                cenx = int(cen["m10"] / cen["m00"])
                ceny = int(cen["m01"] / cen["m00"])
                X = int(cenx / x)
                Y = int(ceny / y)
                block_part[Y][X] = 0
                cv2.drawContours(img[t], cnt, -1, (255, 0, 0), 3)
                cv2.putText(img[t], 'w', (cenx, ceny), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        cv2.imshow(str(t), img[t])

        t = t + 1
        # print(block_part)
        cv2.waitKey(0)
        if block_part[1][1] == 0:
            insert(0, [0, 0, -1], [0, 0, -1])
            insert(block_part[0][0], [0, 0, -1], [1, -1, -1])
            insert(block_part[0][1], [0, 0, -1], [1, 0, -1])
            insert(block_part[0][2], [0, 0, -1], [1, 1, -1])
            insert(block_part[1][0], [0, 0, -1], [0, -1, -1])
            insert(block_part[1][2], [0, 0, -1], [0, 1, -1])
            insert(block_part[2][0], [0, 0, -1], [-1, -1, -1])
            insert(block_part[2][1], [0, 0, -1], [-1, 0, -1])
            insert(block_part[2][2], [0, 0, -1], [-1, 1, -1])
        elif block_part[1][1] == 1:
            insert(1, [1, 0, 0], [1, 0, 0])
            insert(block_part[0][0], [1, 0, 0], [1, -1, 1])
            insert(block_part[0][1], [1, 0, 0], [1, 0, 1])
            insert(block_part[0][2], [1, 0, 0], [1, 1, 1])
            insert(block_part[1][0], [1, 0, 0], [1, -1, 0])
            insert(block_part[1][2], [1, 0, 0], [1, 1, 0])
            insert(block_part[2][0], [1, 0, 0], [1, -1, -1])
            insert(block_part[2][1], [1, 0, 0], [1, 0, -1])
            insert(block_part[2][2], [1, 0, 0], [1, 1, -1])
        elif block_part[1][1] == 2:
            insert(2, [0, 1, 0], [0, 1, 0])
            insert(block_part[0][0], [0, 1, 0], [1, 1, 1])
            insert(block_part[0][1], [0, 1, 0], [0, 1, 1])
            insert(block_part[0][2], [0, 1, 0], [-1, 1, 1])
            insert(block_part[1][0], [0, 1, 0], [1, 1, 0])
            insert(block_part[1][2], [0, 1, 0], [-1, 1, 0])
            insert(block_part[2][0], [0, 1, 0], [1, 1, -1])
            insert(block_part[2][1], [0, 1, 0], [0, 1, -1])
            insert(block_part[2][2], [0, 1, 0], [-1, 1, -1])
        elif block_part[1][1] == 3:
            insert(3, [0, -1, 0], [0, -1, 0])
            insert(block_part[0][0], [0, -1, 0], [-1, -1, 1])
            insert(block_part[0][1], [0, -1, 0], [0, -1, 1])
            insert(block_part[0][2], [0, -1, 0], [1, -1, 1])
            insert(block_part[1][0], [0, -1, 0], [-1, -1, 0])
            insert(block_part[1][2], [0, -1, 0], [1, -1, 0])
            insert(block_part[2][0], [0, -1, 0], [-1, -1, -1])
            insert(block_part[2][1], [0, -1, 0], [0, -1, -1])
            insert(block_part[2][2], [0, -1, 0], [1, -1, -1])
        elif block_part[1][1] == 4:
            insert(4, [-1, 0, 0], [-1, 0, 0])
            insert(block_part[0][0], [-1, 0, 0], [-1, 1, 1])
            insert(block_part[0][1], [-1, 0, 0], [-1, 0, 1])
            insert(block_part[0][2], [-1, 0, 0], [-1, -1, 1])
            insert(block_part[1][0], [-1, 0, 0], [-1, 1, 0])
            insert(block_part[1][2], [-1, 0, 0], [-1, -1, 0])
            insert(block_part[2][0], [-1, 0, 0], [-1, 1, -1])
            insert(block_part[2][1], [-1, 0, 0], [-1, 0, -1])
            insert(block_part[2][2], [-1, 0, 0], [-1, -1, -1])

        elif block_part[1][1] == 5:
            insert(5, [0, 0, 1], [0, 0, 1])
            insert(block_part[0][0], [0, 0, 1], [-1, -1, 1])
            insert(block_part[0][1], [0, 0, 1], [-1, 0, 1])
            insert(block_part[0][2], [0, 0, 1], [-1, 1, 1])
            insert(block_part[1][0], [0, 0, 1], [0, -1, 1])
            insert(block_part[1][2], [0, 0, 1], [0, 1, 1])
            insert(block_part[2][0], [0, 0, 1], [1, -1, 1])
            insert(block_part[2][1], [0, 0, 1], [1, 0, 1])
            insert(block_part[2][2], [0, 0, 1], [1, 1, 1])
    # print(block)
    for iX in range(3):
        for jY in range(3):
            for kZ in range(3):
                i = iX
                j = jY
                k = kZ
                if i == 2:
                    i = -1
                if j == 2:
                    j = -1
                if k == 2:
                    k = -1
                if block[i][j][k] == None:
                    continue
                elif block[i][j][k][1] == -1:
                    face_M[block[i][j][k][0]][0][0] = [i, j, k]
                    face_M[block[i][j][k][0]][0][1] = block[i][j][k][3]
                elif block[i][j][k][2] == -1:
                    a1, a1c, b1, b1c = find2([i, j, k])
                    face_M[a1[0]][a1[1]][0] = [i, j, k]
                    face_M[a1[0]][a1[1]][1] = a1c
                    face_M[b1[0]][b1[1]][0] = [i, j, k]
                    face_M[b1[0]][b1[1]][1] = b1c
                else:
                    a1, a1c, b1, b1c, c1, c1c = find3([i, j, k])
                    face_M[a1[0]][a1[1]][0] = [i, j, k]
                    face_M[a1[0]][a1[1]][1] = a1c
                    face_M[b1[0]][b1[1]][0] = [i, j, k]
                    face_M[b1[0]][b1[1]][1] = b1c
                    face_M[c1[0]][c1[1]][0] = [i, j, k]
                    face_M[c1[0]][c1[1]][1] = c1c

    print("----------------------------------------------------------------------------------------------------------")
    global cntr
    # printfaces()
    # shuffle()

    cntr=0
    global steps
    steps = 0
    print("----------------------------------------------------------------------------------------------------------")
    _ = printfaces()

    bplus()
    print("----------------------------------------------------------------------------------------------------------")
    _ = printfaces()
    f2l()
    print("steps are ", steps)
    print('total steps after optimization are ', cntr)
    print("----------------------------------------------------------------------------------------------------------")
    allface = printfaces()
    topcross(allface[5])
    allface = printfaces()
    oll(allface)
    print("----------------------------------------------------------------------------------------------------------")
    allface = printfaces()
    topcorners(allface)
    print("----------------------------------------------------------------------------------------------------------")
    allface = printfaces()
    topedge(allface)
    print("----------------------------------------------------------------------------------------------------------")
    allface = printfaces()
    print('total steps are', steps)
    print('total steps after optimization are ', cntr)

    fsteps=[]
    for i in stepa:
        if i==1:
            fsteps.append('RC')
        elif i==-1:
            fsteps.append('RA')
        elif i==2:
            fsteps.append('GC')
        elif i==-2:
            fsteps.append('GA')
        elif i==3:
            fsteps.append('BC')
        elif i==-3:
            fsteps.append('BA')
        elif i==4:
            fsteps.append('OC')
        elif i==-4:
            fsteps.append('OA')
        elif i==5:
            fsteps.append('YC')
        elif i==-5:
            fsteps.append('YA')
        elif i==6:
            fsteps.append('WC')
        elif i==-6:
            fsteps.append('WA')
        elif i==None:
            break
    # print(stepa)
    print(fsteps)
    print(len(fsteps))







