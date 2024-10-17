class IntMod:
    def __init__(self, value, mod):
        self.value = value
        self.mod = mod

    def setValue(self, val):
        if 0 <= val <= self.mod:
            self.value = val
        else:
            raise ValueError(f"Value must be between 0 and {self.mod}")

    def add(self, value):
        for i in range(value):
            if self.value == self.mod-1:
                self.value = 0
            else:
                self.value += 1

    def subtract(self, value):
        for i in range(value):
            if self.value == 0:
                self.value = self.mod - 1
            else:
                self.value -= 1
