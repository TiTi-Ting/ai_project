import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from snakeClass import Snake
from snakeAgent import enemyAgent, Snake_num

#------获取特征-------
#目标：将特征[己方蛇头所在坐标、对手蛇头所在坐标、对手身体所在坐标、食物所在坐标]输入3层的神经网络卷积，输出相应的动作，以执行该动作的得分作为reward。
#获取特征实际方法1；蛇头和食物的相对x坐标和y坐标，蛇头上、下、左、右是否有自身身体或者游戏边界作为state，效果很好，训练后AI超过普通玩家水平
#state=[xfood−xhead,yfood−yhead,k1,k2,k3,k4]
def get_state(snake_head, food_position, snake_body, grid_size):
    # 计算相对坐标差值
    relative_x = food_position[0] - snake_head[0]
    relative_y = food_position[1] - snake_head[1]

    # 初始化方向上是否有障碍的标志
    obstacles = [0, 0, 0, 0]  # 上、下、左、右

    # 检查上方是否有障碍
    if snake_head[1] == 0 or (snake_head[0], snake_head[1] - 1) in snake_body:
        obstacles[0] = 1

    # 检查下方是否有障碍
    if snake_head[1] == grid_size[1] - 1 or (snake_head[0], snake_head[1] + 1) in snake_body:
        obstacles[1] = 1

    # 检查左方是否有障碍
    if snake_head[0] == 0 or (snake_head[0] - 1, snake_head[1]) in snake_body:
        obstacles[2] = 1

    # 检查右方是否有障碍
    if snake_head[0] == grid_size[0] - 1 or (snake_head[0] + 1, snake_head[1]) in snake_body:
        obstacles[3] = 1

    # 归一化相对坐标差值
    normalized_relative_x = relative_x / grid_size[0]
    normalized_relative_y = relative_y / grid_size[1]

    # 构建状态向量
    state = [normalized_relative_x, normalized_relative_y] + obstacles

    return np.array(state)

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 输入层，6个特征
        self.fc1 = nn.Linear(6, 64)
        # 隐藏层
        self.fc2 = nn.Linear(64, 32)
        # 输出层，4个动作
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        # 输入层
        x = F.relu(self.fc1(x))
        # 隐藏层
        x = F.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 生成食物
def generate_food(dis_width, dis_height, border_size, player_body, snakes):
    food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    # 确保食物的位置不与玩家蛇和其他蛇的位置重合
    while any(food_pos in player_body or (food_pos in snake.body and snake.alive for snake in snakes)):
        food_pos = [random.randrange(border_size, (dis_width - border_size) // 10) * 10,
                    random.randrange(border_size, (dis_height - border_size) // 10) * 10]

    return food_pos


def generate_snakes(dis_width, dis_height, border_size, snake_body, snake_count):
    snakes = []

    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 128, 0),
              (128, 0, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128),
              (128, 128, 128), (0, 0, 128), (128, 0, 0), (0, 128, 0)]  # 更多颜色

    for _ in range(snake_count):
        #snake_color = random.choice(colors)
        #colors.remove(snake_color)  # 避免重复使用颜色

        snake_color=(169, 169, 169) # grey
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

def get_reward(player, snakes, food_pos, game_over, dis_width, dis_height, border_size):
    def calculate_reward(player, snakes, food_pos, game_over, dis_width, dis_height, border_size):
        reward = 0

        if game_over:
            return -100  # Large negative reward for losing the game

        player_head = player.pos

        # Reward for being closer to food (using reciprocal of distance)
        food_distance = euclidean_distance(player_head, food_pos)
        if food_distance != 0:
            reward += 1 / food_distance  

        # Penalty for being too close to enemies
        for snake in snakes:
            if snake != player:
                enemy_distance = min(euclidean_distance(player_head, snake.pos),
                                     min(euclidean_distance(player_head, segment) for segment in snake.body))
                if enemy_distance != 0:
                    reward -= 1 / enemy_distance 

        # Small penalty for being close to boundaries
        boundary_distance = min(player_head[0] - border_size, dis_width - border_size - player_head[0],
                                player_head[1] - border_size, dis_height - border_size - player_head[1])
        if boundary_distance != 0:
            reward -= 1 / (2 * boundary_distance)

        return reward

    def euclidean_distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    # ans_reward=torch.tensor(10.0)
    ans_reward = calculate_reward(player, snakes, food_pos, game_over, dis_width, dis_height, border_size)
    return torch.tensor(ans_reward).float()
    # return torch.randn(1).float()

def one_step(snake_pos, food_pos, snake_body, snakes, score, game_over):   
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
    for segment in snake_body[1:]:
        if snake_pos[0] == segment[0] and snake_pos[1] == segment[1]:
            game_over = True

    # 判断是否与其他蛇相撞
    for other_snake in snakes:
        if not other_snake.alive:
            continue
        if snake_body[0][0] == other_snake.pos[0] and snake_body[0][1] == other_snake.pos[1]:
            game_over = True
        for segment in other_snake.body:
            if snake_body[0][0] == segment[0] and snake_body[0][1] == segment[1]:
                game_over = True
    return game_over, score
    # if (game_over):
    #     break
    
def move(player, snakes, snake_body, food_pos, score, action):
    all_snakes = snakes.copy()
    all_snakes.append(player)
    # Move all snakes
    for snake in all_snakes:
        if snake == player:
            movepos = action
        else:
            myAgent = enemyAgent() 
            movepos = myAgent(snake, all_snakes, border_size, dis_width - border_size, border_size, dis_height - border_size, food_pos)

        if movepos[0] + snake.dir[0] == 0 and movepos[1] + snake.dir[1] == 0:
            movepos = snake.dir
        else:
            snake.dir = movepos
        snake.pos[0] += movepos[0] * 10
        snake.pos[1] += movepos[1] * 10
        snake.body.insert(0, list(snake.pos))

    # 检测食物
    snakes, food_pos = check_food(snakes, food_pos, snake_body)

    # 检测碰撞
    snakes, addscore = check_collision(snakes, snake_body)
    snakes = check_boundary_collision(snakes)

    score += addscore
    return snakes, score

directions = [(0, 1), (0, -1), (-1, 0), (1, 0)] # up, down, left, right
direction_to_index = {direction: idx for idx, direction in enumerate(directions)}
def train(epochs, model, target_model, optimizer, criterion, replay_buffer, dis_width, dis_height, border_size, batch_size=32, gamma=0.99, target_update_frequency=10, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9, lr=0.0005):
    losses, eps = [], []
    for epoch in range(epochs):
        total_loss = 0
        # reset game
        snake_pos = [dis_width / 2, dis_height / 2]
        snake_body = [[snake_pos[0], snake_pos[1]],
                  [snake_pos[0] - 10, snake_pos[1]],
                  [snake_pos[0] - 20, snake_pos[1]]]
        score = 0
        direction = (1, 0)
        snakes = generate_snakes(dis_width, dis_height, border_size, snake_body, Snake_num())
        food_pos = generate_food(dis_width, dis_height, border_size, snake_body, snakes)
        game_over = False
        
        while True:
            ## decide action
            state = get_state((snake_pos[0], snake_pos[1]), food_pos, snake_body, (dis_width, dis_height))
            state_tensor = torch.from_numpy(state)
            input_data = torch.unsqueeze(state_tensor, 0).float()

            # ε-greedy strategy for action selection
            if random.random() > epsilon:
                output = model(input_data)
                action_index = torch.argmax(output, dim=1).item()
                action = directions[action_index]  # map to action

            else:
                action = random.choice(directions)  # random action
                action_index = direction_to_index[action]

            # Perform action and get next state and reward
            game_over, score = one_step(snake_pos, food_pos, snake_body, snakes, score, game_over)
            if (game_over):
                break

            player = Snake((0, 255, 0), snake_pos, direction, snake_body, score, True)
            snakes, score = move(player, snakes, snake_body, food_pos,score, action)
            
            next_state = get_state((snake_pos[0], snake_pos[1]), food_pos, snake_body, (dis_width, dis_height))
            reward = get_reward(player, snakes, food_pos, game_over, dis_width, dis_height, border_size)
            done = game_over

            # Store experience in replay buffer
            replay_buffer.push(state, action_index, reward, next_state, float(done))


            # Check if replay buffer is large enough for sampling
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                for state, action_index, reward, next_state, done in batch:
                    state = torch.FloatTensor(state).unsqueeze(0)  # Ensure this is a 2D tensor
                    next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Ensure this is a 2D tensor
                    action = torch.LongTensor([action_index])  # Ensure this is a tensor
                    reward = torch.FloatTensor([reward])
                    done = torch.FloatTensor([done])

                    current_q_values = model(state)
                    current_q = current_q_values[0, action_index].unsqueeze(0) 

                    next_state_q_values = target_model(next_state)
                    max_next_state_q_values = torch.max(next_state_q_values)  
                    target_q = reward + gamma * max_next_state_q_values * (1 - done)
                    # print(current_q, target_q)
                    
                    # Compute loss
                    loss = criterion(current_q, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            if done:
                break
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        losses.append(total_loss)
        eps.append(epsilon)
        # Update target network
        if epoch % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())
            
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'model_ep{epoch}_{lr}.pth')
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, Epsilon: {epsilon}")

    return model, losses, eps

dis_width = 820
dis_height = 820
border_size = 20

# Initialize models, optimizer, loss function, and replay buffer
learning_rate = 0.0002 # 0.0005
model = CustomNet()
target_model = CustomNet()
target_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer(capacity=10000)

# Call the train function with necessary parameters
trained_model, losses, eps = train(epochs=51, model=model, target_model=target_model, optimizer=optimizer, 
                      criterion=criterion, replay_buffer=replay_buffer, dis_width=500, dis_height=500, 
                      border_size=10, lr=learning_rate)

print(losses, eps)

