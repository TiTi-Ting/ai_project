import pygame
import time
import random
from snakeClass import Snake
from snakeAgent import Agent, Snake_num


def move_snakes(snakes, player, x1, x2, y1, y2, foodpos):
    all_snake = snakes.copy()
    all_snake.append(player)

    for snake in snakes:
        myAgent = Agent()
        movepos = myAgent(snake, all_snake, x1, x2, y1, y2, foodpos)
        if movepos[0] + snake.dir[0] == 0 and movepos[1] + snake.dir[1] == 0:
            movepos = snake.dir
        else:
            snake.dir = movepos
        snake.pos[0] += movepos[0] * 10
        snake.pos[1] += movepos[1] * 10
        snake.body.insert(0, list(snake.pos))
    return snakes


def generate_snakes(dis, dis_width, dis_height, border_size, snake_body, snake_count):
    snakes = []

    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 128, 0),
              (128, 0, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128),
              (128, 128, 128), (0, 0, 128), (128, 0, 0), (0, 128, 0)]  # 更多颜色

    for _ in range(snake_count):
        #snake_color = random.choice(colors)
        #colors.remove(snake_color)  # 避免重复使用颜色

        snake_color=grey
        snake_pos = [random.randrange(border_size, ((dis_width - border_size) // 10)) * 10 - 20,
                     random.randrange(border_size, ((dis_height - border_size) // 10)) * 10 - 20]

        # 确保新蛇的位置不和其他蛇的初始位置重叠
        while abs(snake_pos[0] - dis_width // 2) < 50 and abs(snake_pos[1] - dis_height // 2) < 50 and (any(abs(snake_pos[0] - other_snake.pos[0]) < 30 and abs(snake_pos[1] - other_snake.pos[1]) < 30 for other_snake in snakes)):
            snake_pos = [random.randrange(border_size, ((dis_width - border_size) // 10)) * 10,
                         random.randrange(border_size, ((dis_height - border_size) // 10)) * 10]

        snake_body = [snake_pos.copy(), [snake_pos[0] - 10, snake_pos[1]],
                      [snake_pos[0] - 20, snake_pos[1]]]

        snake = Snake(color=snake_color, pos=snake_pos, dir=(1,0), 
                      body=snake_body, score=0, alive=True)

        snakes.append(snake)

    return snakes

# 检测蛇是否吃到食物
def check_food(snakes, food_pos, player_body):
    isFoodEat = False
    for snake in snakes:
        if not snake.alive:
            continue        
        if snake.pos[0] == food_pos[0] and snake.pos[1] == food_pos[1]:
            # 吃到食物，增加积分
            snake.score += 1
            isFoodEat = True
        else:
            snake.body.pop()

    if isFoodEat:
        food_pos = generate_food(
            dis_width, dis_height, border_size, player_body, snakes)
    return (snakes,food_pos)

# 检测蛇是否碰到其他蛇
def check_collision(snakes, player_body):
    score = 0
    die_list = []

    # 撞到玩家
    for snake in snakes:
        if not snake.alive:
            continue
        if (snake.pos in player_body):
            score += snake.score
            die_list.append(snake)
    # 撞到其他蛇
    for i in range(len(snakes)):
        if not snakes[i].alive:
            continue
        for j in range(i + 1, len(snakes)):
            if not snakes[j].alive:
                continue
            if snakes[i].pos == snakes[j].pos or snakes[i].pos in snakes[j].body:
                # 蛇i碰到了蛇j，蛇i死亡
                snakes[j].score += snakes[i].score
                die_list.append(snakes[i])

    for snake in die_list:
        snake.alive = False
        snake.score = 0

    return (snakes, score)

# 检测蛇是否碰到边界
def check_boundary_collision(snakes):
    for snake in snakes:
        if not snake.alive:
            continue
        if snake.pos[0] < border_size or snake.pos[0] >= dis_width - border_size or \
           snake.pos[1] < border_size or snake.pos[1] >= dis_height - border_size:
            # 蛇碰到边界，蛇死亡
            snake.alive = False
            snake.score = 0
    return snakes

# 绘制其他蛇
def draw_snakes(dis, snakes):
    for snake in snakes:
        if not snake.alive:
            continue
        if snake.alive:
            pygame.draw.rect(dis, snake.color, [
                             snake.pos[0], snake.pos[1], 10, 10])
            for segment in snake.body:
                pygame.draw.rect(dis, snake.color, [
                                 segment[0], segment[1], 10, 10])
                
# 绘制边框函数
def draw_border(dis, dis_width, dis_height, border_size):
    border_color = grey
    pygame.draw.rect(dis, border_color, [0, 0, dis_width, border_size])  # 上边框
    pygame.draw.rect(dis, border_color, [
                     0, dis_height - border_size, dis_width, border_size])  # 下边框
    pygame.draw.rect(dis, border_color, [0, 0, border_size, dis_height])  # 左边框
    pygame.draw.rect(dis, border_color, [
                     dis_width - border_size, 0, border_size, dis_height])  # 右边框

# 生成食物
def generate_food(dis_width, dis_height, border_size, player_body, snakes):
    food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    # 确保食物的位置不与玩家蛇和其他蛇的位置重合
    while any(food_pos in player_body or (food_pos in snake.body and snake.alive for snake in snakes)):
        food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                    random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    return food_pos


# 显示游戏结束信息和重新开始按钮
def show_game_over(dis, dis_width, dis_height):
    game_over_font = pygame.font.SysFont(None, 80)
    game_over_message = game_over_font.render("Game Over!", True, (255, 0, 0))
    dis.blit(game_over_message, [dis_width // 3 - 20, dis_height / 3])

    restart_font = pygame.font.SysFont(None, 50)
    restart_message = restart_font.render(
        "Press R to Restart", False, (255, 255, 255))
    dis.blit(restart_message, [dis_width // 3 - 20, dis_height / 2])

    pygame.display.flip()

# 开始游戏
def game_start(dis, dis_width, dis_height, border_size, snake_speed):
    game_loop(dis, dis_width, dis_height, border_size, snake_speed)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if (event.key == pygame.K_r):
                    game_loop(dis, dis_width, dis_height,
                              border_size, snake_speed)


# 显示得分
def show_score(dis, score):
    value = font_style.render("Your Score: " + str(score), True, white)
    dis.blit(value, [0, 0])

# 定义开始界面函数
def game_intro(dis, dis_width, dis_height):
    intro = True
    snake_speed = None

    large_font = pygame.font.SysFont(None, 80)
    small_font = pygame.font.SysFont(None, 50)

    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    snake_speed = 6
                    intro = False
                elif event.key == pygame.K_2:
                    snake_speed = 11
                    intro = False
                elif event.key == pygame.K_3:
                    snake_speed = 20
                    intro = False

        dis.fill(black)

        title_text = large_font.render("Competitive Greedy Snake", True, white)
        dis.blit(title_text, [dis_width // 2 -
                 title_text.get_width() // 2, dis_height // 4])

        easy_text = small_font.render("1 EASY", True, green)
        medium_text = small_font.render("2 MEDIUM", True, yellow)
        hard_text = small_font.render("3 HARD", True, red)

        dis.blit(easy_text, [dis_width // 2 -
                 easy_text.get_width() // 2, dis_height // 2])
        dis.blit(medium_text, [
                 dis_width // 2 - medium_text.get_width() // 2, dis_height // 2 + 50])
        dis.blit(hard_text, [dis_width // 2 -
                 hard_text.get_width() // 2, dis_height // 2 + 100])

        pygame.display.update()

    return snake_speed

# 定义游戏主循环函数
def game_loop(dis, dis_width, dis_height, border_size, snake_speed):
    snake_block = 10

    snake_pos = [dis_width / 2, dis_height / 2]
    snake_body = [[snake_pos[0], snake_pos[1]],
                  [snake_pos[0] - 10, snake_pos[1]],
                  [snake_pos[0] - 20, snake_pos[1]]]

    direction = RIGHT
    change_to = direction

    score = 0
    game_over = False

    snakes = generate_snakes(dis, dis_width, dis_height,
                             border_size, snake_body, Snake_num())

    food_pos = generate_food(dis_width, dis_height,
                             border_size, snake_body, snakes)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    change_to = UP
                elif event.key == pygame.K_DOWN:
                    change_to = DOWN
                elif event.key == pygame.K_LEFT:
                    change_to = LEFT
                elif event.key == pygame.K_RIGHT:
                    change_to = RIGHT

        if change_to == UP and not direction == DOWN:
            direction = UP
        if change_to == DOWN and not direction == UP:
            direction = DOWN
        if change_to == LEFT and not direction == RIGHT:
            direction = LEFT
        if change_to == RIGHT and not direction == LEFT:
            direction = RIGHT

        if direction == UP:
            snake_pos[1] -= snake_block
        if direction == DOWN:
            snake_pos[1] += snake_block
        if direction == LEFT:
            snake_pos[0] -= snake_block
        if direction == RIGHT:
            snake_pos[0] += snake_block

        snake_body.insert(0, list(snake_pos))

        if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
            score += 1
            food_pos = generate_food(
                dis_width, dis_height, border_size, snake_body, snakes)
        else:
            snake_body.pop()

        # 判断是否撞墙或者撞到自己
        if snake_pos[0] < border_size or snake_pos[0] >= dis_width - border_size or \
           snake_pos[1] < border_size or snake_pos[1] >= dis_height - border_size:
            game_over = True
            show_game_over(dis, dis_width, dis_height)
        for segment in snake_body[1:]:
            if snake_pos[0] == segment[0] and snake_pos[1] == segment[1]:
                game_over = True
                show_game_over(dis, dis_width, dis_height)

        # 判断是否与其他蛇相撞
        for other_snake in snakes:
            if not other_snake.alive:
                continue
            if snake_body[0][0] == other_snake.pos[0] and snake_body[0][1] == other_snake.pos[1]:
                game_over = True
                show_game_over(dis, dis_width, dis_height)
            for segment in other_snake.body:
                if snake_body[0][0] == segment[0] and snake_body[0][1] == segment[1]:
                    game_over = True
                    show_game_over(dis, dis_width, dis_height)

        if (game_over):
            break

        player = Snake(green, snake_pos, direction, snake_body, score, True)
        snakes = move_snakes(snakes, player, border_size, dis_width -
                             border_size, border_size, dis_height - border_size, food_pos)

        # 检测食物
        snakes, food_pos = check_food(snakes, food_pos, snake_body)

        # 检测碰撞
        snakes, addscore = check_collision(snakes, snake_body)
        snakes = check_boundary_collision(snakes)

        score += addscore
        
        dis.fill(black)
        draw_snakes(dis, snakes)
        draw_border(dis, dis_width, dis_height, border_size)

        for segment in snake_body:
            pygame.draw.rect(dis, green, pygame.Rect(
                segment[0], segment[1], snake_block, snake_block))

        pygame.draw.rect(dis, yellow, pygame.Rect(
            food_pos[0], food_pos[1], snake_block, snake_block))

        show_score(dis, score)

        pygame.display.flip()

        pygame.time.Clock().tick(snake_speed)


pygame.init()
# 设置颜色
white = (255, 255, 255)
yellow = (255, 255, 102)
green = (0, 255, 0)
black = (0, 0, 0)
red = (213, 50, 80)
grey = (169, 169, 169)

# 定义字体和字号
font_style = pygame.font.SysFont(None, 50)

# 定义方向
UP = 'UP'
DOWN = 'DOWN'
LEFT = 'LEFT'
RIGHT = 'RIGHT'

# 主程序

dis_width = 820
dis_height = 820
border_size = 20
dis = pygame.display.set_mode((dis_width, dis_height))
pygame.display.set_caption('Competitive Greedy Snake')

snake_speed = game_intro(dis, dis_width, dis_height)
game_start(dis, dis_width, dis_height, border_size, snake_speed)

pygame.quit()
quit()
