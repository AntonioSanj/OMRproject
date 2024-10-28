class NoteIndex:
    def __init__(self, value, mod):
        self.value = value
        self.mod = mod

    def setValue(self, val):
        if 1 <= val <= self.mod:
            self.value = val
        else:
            raise ValueError(f"Value must be between 1 and {self.mod}")

    def add(self, value):
        for i in range(value):
            if self.value == self.mod:
                self.value = 1
            else:
                self.value += 1

    def subtract(self, value):
        for i in range(value):
            if self.value == 1:
                self.value = self.mod
            else:
                self.value -= 1
