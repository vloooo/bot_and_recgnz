import cv2
import math


class PossibleChar:

    def __init__(self, _contour):
        self.contour = _contour

        self.bounding_rect = cv2.boundingRect(self.contour)

        [int_x, int_y, int_width, int_height] = self.bounding_rect

        self.pos_x = int_x
        self.pos_y = int_y
        self.width = int_width
        self.height = int_height

        self.area = self.width * self.height

        self.center_x = (self.pos_x + self.pos_x + self.width) / 2
        self.center_y = (self.pos_y + self.pos_y + self.height) / 2

        self.diagonal_size = math.sqrt((self.width ** 2) + (self.height ** 2))

        self.aspect_ratio = float(self.width) / float(self.height)
