# Snake结构体
class Snake:
    def __init__(self, color, pos, dir, body, score, alive):
        self.color = color
        self.pos = pos
        self.dir = dir
        self.body = body
        self.score = score
        self.alive = alive
    #读取蛇身体所在位置
    def get_body(self):
        return self.body