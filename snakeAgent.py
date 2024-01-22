from snakeClass import Snake
import random


#敌方蛇的移动
def enemyAgent():
    return directToFoodAgent  # !!!CHANGE THIS!!!

#我方蛇的移动
def myAgent():
    return directToFoodAgent

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
    return random.choice([(1, 0),(-1,0),(0,1),(0,-1)])


def directToFoodAgent(snake, player, x1, x2, y1, y2, foodpos):
    if not snake or not foodpos:
        return (0, 0)  # 如果蛇为空或者没有食物位置，返回不移动

    head_pos = snake.pos
    food_x, food_y = foodpos

    # 计算水平和垂直方向上的距离
    delta_x = food_x - head_pos[0]
    delta_y = food_y - head_pos[1]

    # 检查是否靠近边界
    near_border_x = head_pos[0] <= x1 + 20 or head_pos[0] >= x2 - 20
    near_border_y = head_pos[1] <= y1 + 20 or head_pos[1] >= y2 - 20

    # 如果靠近边界，选择向食物的方向左转或右转
    if near_border_x and abs(delta_x) >= abs(delta_y):
        if delta_y >= 0:
            direction = (0, 1)  # 向上移动
        else:
            direction = (0, -1)  # 向下移动
    elif near_border_y and abs(delta_y) >= abs(delta_x):
        if delta_x >= 0:
            direction = (1, 0)  # 向左移动
        else:
            direction = (-1, 0)  # 向右移动
    else:
        # 选择靠近食物的方向
        if abs(delta_x) >= abs(delta_y):
            direction = (delta_x // abs(delta_x), 0)  # 水平移动方向
        else:
            direction = (0, delta_y // abs(delta_y))  # 垂直移动方向

    return direction






'''
def someotherAgent(snakes, player, x1, x2, y1, y2, foodpos):
...


'''
