class man:
    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")


if __name__ == '__main__':
    m = man("Davie")
    m.hello()
    m.goodbye()