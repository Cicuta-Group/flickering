import numpy as np

class uc_iter():
    def __init__(self, text):
        self.text = text.upper()
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = self.text[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

class SpiralMoves:
    def __init__(self, move_by, max_fields = 100, start_at = [0, 0], absolute_positions = False, name = None, rotation=0):
        self.move_by = move_by
        self.max_fields = max_fields
        self.start_at = np.array(start_at)
        self.absolute_positions = absolute_positions
        self._length = 1
        self._axis = 0
        self._dir = 1
        self._remaining = self._length
        self._current = np.array(start_at).astype(float)
        self.name = name
        self.index = 0
        self.processed_cells = 0
        self.rotation = rotation

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.max_fields and self.max_fields > 0:
            raise StopIteration

        if self.index == 0:
            self.index += 1
            return self.start_at.copy()

        self._remaining -= 1
        if self._axis == 0:
            shift = self._dir*np.array([self.move_by[0], 0])
        else:
            shift = self._dir*np.array([0, self.move_by[1]])

        if self._remaining == 0:
            #end of the line, switch direction
            if self._axis == 1:
                self._length +=1
                self._remaining = self._length
                self._axis = 0
                self._dir *= -1
            else:
                self._axis = 1
                self._remaining = self._length

        if self.rotation != 0:
            angle = np.deg2rad(self.rotation)
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            shift = np.dot(rotation_matrix, shift)
        self._current += shift

        self.index += 1
        if self.absolute_positions:
            return np.array(self._current).copy()

        return shift

    @property
    def position(self):
        return self._current

    @property
    def field(self):
        return self.index

    def reset(self):
        self.index = 0
        self.processed_cells = 0
        self._dir = 1
        self._current = self.start_at - self._dir*np.array([self.move_by[0], 0]) #avoid move in first call
        self._length = 1
        self._axis = 0
        self._remaining = self._length


if __name__ == "__main__":
    test_spiral = SpiralMoves([100,200], start_at=[2000,3000], absolute_positions = True)
    test_spiral2 = SpiralMoves([100,200], start_at=[-2000,-3000], absolute_positions = True)

    #with this we can go to next after n fields or some time
    for j in range(3):
        i=0
        for shift in test_spiral:
            print(shift)
            i+=1
            if i>3:
                break
        i=0
        for shift in test_spiral2:
            print(shift)
            i+=1
            if i>3:
                break
