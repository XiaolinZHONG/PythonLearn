# -*- coding: utf-8 -*-
# @Time   : 2017/10/17
# @Author : XL ZHONG
# @File   : Tools.py

import math


class Tools:
    def get_distances(self, long1, lat1, long2, lat2):
        r = 6378137  # 地球半径

        lat1 = lat1 * math.pi / 180.0
        lat2 = lat2 * math.pi / 180.0

        a = lat1 - lat2
        b = (long1 - long2) * math.pi / 180.0

        sa2 = math.sin(a / 2.0)
        sb2 = math.sin(b / 2.0)
        d = 2 * r * math.asin(math.sqrt(sa2 * sa2 + math.cos(lat1) * math.cos(lat2) * sb2 * sb2))
        return d

    def get_angles(self, long1, lat1, long2, lat2):
        y = math.sin(long2 - long1) * math.cos(lat2)

        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(long2 - long1)

        b = math.atan2(y, x)

        b = math.degrees(b)
        if b < 0:
            b = b + 360

        return b
