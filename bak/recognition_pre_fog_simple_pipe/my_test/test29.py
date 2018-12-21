
class C1:
    @property
    def f1(self):
        print(4)
        return 5

    def f2(self,x=35):
        print('f2',x)


if __name__ =="__main__":
    c= C1()
    print(c.f1)
    print(c.f1)
