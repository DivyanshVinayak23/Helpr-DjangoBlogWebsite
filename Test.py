class A1:
    def __init__(self):
        self.abc = 111
class B1(A1):
    def __init__(self):
        super().__init__()
        self.pqr = 222
obj = B1()
print(obj.abc)
print(obj.pqr)