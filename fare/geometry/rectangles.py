
class Rectangle(object):
    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)
        self.width = self.right - self.left
        self.height = self.bottom - self.top

    def area(self):
        return self.width * self.height

    def get_center(self):
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def enlarge_bbox(self, enlarge_factor):
        x1 = int(self.left - self.width * (enlarge_factor - 1))
        y1 = int(self.top - self.height * (enlarge_factor - 1))

        x2 = int(self.right + self.width * (enlarge_factor - 1))
        y2 = int(self.bottom + self.height * (enlarge_factor - 1))

        self.left, self.top, self.right, self.bottom = x1, y1, x2, y2
        self.width = self.right - self.left
        self.height = self.bottom - self.top

        return x1, y1, x2, y2

    def translation(self, delta_x, delta_y):
        self.left += int(delta_x)
        self.top += int(delta_y)

        self.right += int(delta_x)
        self.bottom += int(delta_y)
