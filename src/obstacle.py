
class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_position(self):
        return int(self.x), int(self.y)

    def copy(self):
        (x, y) = self.x, self.y
        copy_obs = Obstacle(x, y)
        return copy_obs