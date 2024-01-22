from snakeClass import Snake
import random

def Agent():
    return randomAgent  # !!!CHANGE THIS!!!

def Snake_num():
    return 6 #代表蛇的数量，可修改 最大13
'''

参数解释：

    snakes和player均为Snake类
    以下是Snake的定义
class Snake:
    def __init__(self, color, pos, body, score, alive):
        self.color = color 颜色，不会用到
        self.pos = pos 蛇头部的位置,横坐标pos[0],纵坐标pos[1],向右就是pos[0]+=1,向下就是pos[0]+=1
        self.dir = dir 蛇前进的方向 (±1,0)/(0,±1) ,注意蛇不能反向,所以实际上蛇每步只有三个方向可以选择。
                如果Agent返回了错误的方向,则不会执行该方向
                dir不需要修改,main.py会根据return的方向坐标修改dir
        self.body = body 蛇所有身体的坐标,例如可以用some_pos in some_snake.body来判断某个位置是否会撞到别的蛇
        self.score = score 蛇的得分
        self.alive = alive 蛇是否还活着 True/False 所以访问蛇数组时要先判断是否alive

    x1 x2 y1 y2为地图边界 横坐标[x1,x2] 纵坐标[y1,y2]

    foodpos食物坐标

    当写好Agent后,修改Agent()函数（在最上面！）的返回值为其他函数名以测试

'''
# 敌方agent实现
def keepRightAgent(snakes: Snake, player, x1, x2, y1, y2, foodpos):

    return (1, 0)  # 向右走

def randomAgent(snakes: Snake, player, x1, x2, y1, y2, foodpos):
    #print(snakes)
    #print(player)
    return random.choice([(1, 0),(-1,0),(0,1),(0,-1)])


def directToFoodAgent(snakes, player, x1, x2, y1, y2, foodpos):
    '''
    Need to complete
    '''
    pass


'''
def someotherAgent(snakes, player, x1, x2, y1, y2, foodpos):
...


'''