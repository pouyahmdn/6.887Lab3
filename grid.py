import numpy as np
import copy
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as image

class GridWorld(object):
    def __init__(self, cols=4, rows=3, death_cells=None, coin_cells=None, blocked_cells=None, start_point=None):
        assert cols > 2
        assert rows > 1
        assert cols * rows > 3
        if death_cells is None:
            death_cells = [(cols-1, 1)]
        if coin_cells is None:
            coin_cells = [(cols-1, 0)]
        if blocked_cells is None:
            blocked_cells = [(1, 1)]
        if start_point is None:
            start_point = (0, 1)
        
        # 0: Empty, 1: Blocked, 2: Death, 3: Coin
        self.grid = np.zeros((cols, rows))
        for (x, y), v in list(zip(blocked_cells, [1]*len(blocked_cells))) + \
                    list(zip(death_cells, [2]*len(death_cells))) + \
                    list(zip(coin_cells, [3]*len(coin_cells))):
            assert 0 <= x < cols
            assert 0 <= y < rows
            assert self.grid[x, y] == 0
            self.grid[x, y] = v
        
        self.rew = np.zeros((cols, rows)) - 0.05
        self.rew[self.grid == 2] = -1
        self.rew[self.grid == 3] = 1
        
        assert 0 <= start_point[0] < cols
        assert 0 <= start_point[1] < rows
        self.start_state = [start_point[0], start_point[1]]
        
        self.state = copy.deepcopy(self.start_state)
        self.terminated = False
        
        self.act_to_d = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        self.cell_width = 100
        self.grid_pic = 255 * np.ones((cols*self.cell_width, rows*self.cell_width, 3), dtype=np.uint8)

        self.img_coin = image.imread('coin.png', format='jpeg')
        self.img_death = image.imread('death.png', format='jpeg')

        for x, y in itertools.product(range(cols), range(rows)):
            if self.grid[x, y] == 1:
                self.grid_pic[x*self.cell_width:(x+1)*self.cell_width,
                              y*self.cell_width:(y+1)*self.cell_width] = np.array([90, 90, 90])
    
    def reset(self):
        self.state = copy.deepcopy(self.start_state)
        self.terminated = False
        return np.array(self.state)
    
    def step(self, act):
        # Actions:
        #    0: Right
        #    1: Down
        #    2: Left
        #    3: Up
        assert act in [0, 1, 2, 3]
        assert not self.terminated
        dx, dy = self.act_to_d[act]
        next_x, next_y = self.state[0]+dx, self.state[1]+dy
        next_x, next_y = np.clip(next_x, 0, self.grid.shape[0]-1), np.clip(next_y, 0, self.grid.shape[1]-1)
        if self.grid[next_x, next_y] == 1:
            next_x, next_y = self.state[0], self.state[1]
        if self.grid[next_x, next_y] in [2, 3]:
            self.terminated = True
        
        self.state = [next_x, next_y]
        return np.array(self.state), self.rew[self.state[0], self.state[1]], self.terminated, {}
    
    def _render_base(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()
        ax.clear()
        ax.imshow(self.grid_pic.transpose(1,0,2))
        for x, y in itertools.product(range(self.grid.shape[0]), range(self.grid.shape[1])):
            if self.grid[x, y] == 2:
                box_death = OffsetImage(self.img_death, zoom=0.03)
                ax.add_artist(AnnotationBbox(box_death, ((x+0.7)*self.cell_width, (y+0.3)*self.cell_width), frameon=False))
            if self.grid[x, y] == 3:
                box_coin = OffsetImage(self.img_coin, zoom=0.04)
                ax.add_artist(AnnotationBbox(box_coin, ((x+0.75)*self.cell_width, (y+0.25)*self.cell_width), frameon=False))
        ax.set_xticks([(i+0.5)*self.cell_width for i in range(self.grid.shape[0])])
        ax.set_xticklabels([i for i in range(self.grid.shape[0])])
        ax.set_yticks([(i+0.5)*self.cell_width for i in range(self.grid.shape[1])])
        ax.set_yticklabels([i for i in range(self.grid.shape[1])])
        ax.vlines([i*self.cell_width for i in range(1, self.grid.shape[0])], 0, 1, transform=ax.get_xaxis_transform(), linestyle='--')
        ax.hlines([i*self.cell_width for i in range(1, self.grid.shape[1])], 0, 1, transform=ax.get_yaxis_transform(), linestyle='--')
        return ax
        
    def render(self, ax=None):
        ax = self._render_base(ax)
        circ = Circle(((self.state[0]+0.5)*self.cell_width, (self.state[1]+0.5)*self.cell_width), 25, zorder=10)
        ax.add_patch(circ)
        return ax
    
    def visualize_qtable(self, qtable, ax=None):
        assert qtable.shape == (self.grid.shape[0], self.grid.shape[1], 4)
        ax = self._render_base(ax)
        for x, y in itertools.product(range(self.grid.shape[0]), range(self.grid.shape[1])):
            if self.grid[x, y] != 0:
                continue
            dir_opt = np.argmax(qtable[x, y])
            dx, dy = self.act_to_d[dir_opt]
            p = FancyArrowPatch(((x+0.5-dx*0.3)*self.cell_width, (y+0.5-dy*0.3)*self.cell_width), ((x+0.5+dx*0.3)*self.cell_width, (y+0.5+dy*0.3)*self.cell_width), arrowstyle='->', mutation_scale=12, linewidth=3)
            ax.add_patch(p)
        return ax
    
    def visualize_policy_table(self, policy_table, ax=None):
        assert policy_table.shape == (self.grid.shape[0], self.grid.shape[1], 4)
        ax = self._render_base(ax)
        for x, y in itertools.product(range(self.grid.shape[0]), range(self.grid.shape[1])):
            if self.grid[x, y] != 0:
                continue
            for act in range(4):
                dx, dy = self.act_to_d[act]
                p = FancyArrowPatch(((x+0.5)*self.cell_width, (y+0.5)*self.cell_width), ((x+0.5+dx*0.4*(policy_table[x, y, act]+0.05))*self.cell_width, (y+0.5+dy*(0.4*policy_table[x, y, act]+0.05))*self.cell_width), arrowstyle='->', mutation_scale=6, linewidth=1)
                ax.add_patch(p)
        return ax