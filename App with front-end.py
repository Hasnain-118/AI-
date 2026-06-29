import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import heapq
import time

MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))

def neighbors(grid, cell):
    rows, cols = len(grid), len(grid[0])
    r, c = cell
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 1:
            yield nr, nc

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(parent, start, goal):
    path, current = [], goal
    while current != start:
        path.append(current)
        current = parent[current[0]][current[1]]
        if current is None:
            return []
    path.append(start)
    path.reverse()
    return path

def run_search(grid, start, goal, mode):
    rows, cols = len(grid), len(grid[0])
    parent = [[None] * cols for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]
    explored = []
    frontier_order = [start]
    frontier_set = {start}

    def push_frontier(cell):
        if cell not in frontier_set:
            frontier_order.append(cell)
            frontier_set.add(cell)

    def drop_frontier(cell):
        frontier_set.discard(cell)
        if cell in frontier_order:
            frontier_order.remove(cell)

    uses_heap = mode in ("greedy", "astar")
    if mode == "bfs":
        queue = deque([start])
        pop = queue.popleft
    elif mode == "dfs":
        queue = [start]
        pop = queue.pop
    else:
        g_cost = [[float("inf")] * cols for _ in range(rows)]
        g_cost[start[0]][start[1]] = 0
        weight = {"greedy": lambda g, c: manhattan(c, goal),
                  "astar": lambda g, c: g + manhattan(c, goal)}[mode]
        heap = [(weight(0, start), start)]
        pop = lambda: heapq.heappop(heap)[1]

    if not uses_heap:
        visited[start[0]][start[1]] = True

    while (queue if not uses_heap else heap):
        current = pop()

        if uses_heap:
            if visited[current[0]][current[1]]:
                continue
            visited[current[0]][current[1]] = True

        drop_frontier(current)
        explored.append(current)

        if current == goal:
            yield frontier_order[:], explored, current, reconstruct_path(parent, start, goal)
            return

        for nxt in neighbors(grid, current):
            nr, nc = nxt
            if not uses_heap:
                if not visited[nr][nc]:
                    visited[nr][nc] = True
                    parent[nr][nc] = current
                    queue.append(nxt)
                    push_frontier(nxt)
            elif mode == "greedy":
                if not visited[nr][nc]:
                    parent[nr][nc] = current
                    heapq.heappush(heap, (weight(0, nxt), nxt))
                    push_frontier(nxt)
            else:
                new_g = g_cost[current[0]][current[1]] + 1
                if new_g < g_cost[nr][nc]:
                    g_cost[nr][nc] = new_g
                    parent[nr][nc] = current
                    heapq.heappush(heap, (weight(new_g, nxt), nxt))
                    push_frontier(nxt)

        yield frontier_order[:], explored, current, None

    yield [], explored, None, None

ALGORITHMS = {
    "BFS": "bfs",
    "DFS": "dfs",
    "Greedy": "greedy",
    "A*": "astar",
}
ALGORITHM_NAMES = list(ALGORITHMS.keys())

class SearchVisualizer:
    CELL = 30
    ROWS = 20
    COLS = 24

    EMPTY, WALL, START, GOAL, FRONTIER, EXPLORED, PATH = range(7)

    def __init__(self, root):
        self.root = root
        self.root.title("Pathfinding Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        self.root.resizable(False, False)

        self.grid = [[self.EMPTY] * self.COLS for _ in range(self.ROWS)]
        self.start = None
        self.goal = None
        self.mode = "start"
        self.algorithm = "BFS"
        self.speed = 30
        self.is_running = False
        self.anim_job = None
        self.stats = []
        self.comparing = False
        self.compare_generator = None

        self.colors = {
            "bg": "#1a1a1a",
            "panel": "#2d2d2d",
            "header": "#ffffff",
            "text": "#e0e0e0",
            "muted": "#888888",
            "border": "#3d3d3d",
            "empty": "#252525",
            "wall": "#4a4a4a",
            "start": "#00b894",
            "goal": "#e17055",
            "frontier": "#74b9ff",
            "explored": "#2d2d2d",
            "path": "#fdcb6e",
            "accent": "#6c5ce7",
            "success": "#00b894",
            "danger": "#e17055",
        }

        self._build_layout()
        self._redraw()

    def _build_layout(self):
        main = tk.Frame(self.root, bg=self.colors["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self._build_header(main)
        self._build_controls(main)
        self._build_legend(main)
        
        content = tk.Frame(main, bg=self.colors["bg"])
        content.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        left = tk.Frame(content, bg=self.colors["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        canvas_box = tk.Frame(left, bg=self.colors["border"], padx=2, pady=2)
        canvas_box.pack(expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(canvas_box, 
                               width=self.COLS * self.CELL,
                               height=self.ROWS * self.CELL,
                               bg=self.colors["empty"], 
                               highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<Button-3>", self.on_erase_wall)
        
        right = tk.Frame(content, bg=self.colors["bg"], width=350)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(15, 0))
        right.pack_propagate(False)
        
        self._build_results(right)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=self.colors["bg"])
        header.pack(fill=tk.X, pady=(0, 10))

        title = tk.Label(header, text="PATHFINDING VISUALIZER", 
                        font=("Courier New", 18, "bold"),
                        bg=self.colors["bg"], fg=self.colors["header"])
        title.pack(side=tk.LEFT)

        self.status = tk.StringVar(value="[ READY ] Place start node")
        status = tk.Label(header, textvariable=self.status, 
                         font=("Courier New", 10),
                         bg=self.colors["bg"], fg=self.colors["muted"])
        status.pack(side=tk.RIGHT)

    def _build_controls(self, parent):
        box = tk.Frame(parent, bg=self.colors["panel"], padx=10, pady=8)
        box.pack(fill=tk.X, pady=(0, 8))

        left = tk.Frame(box, bg=self.colors["panel"])
        left.pack(side=tk.LEFT)

        tk.Label(left, text="ALGORITHM:", font=("Courier New", 9, "bold"),
                bg=self.colors["panel"], fg=self.colors["muted"]).pack(side=tk.LEFT, padx=(0, 5))

        self.algo_var = tk.StringVar(value="BFS")
        algo = ttk.Combobox(left, textvariable=self.algo_var, 
                           values=ALGORITHM_NAMES, state="readonly", 
                           width=8, font=("Courier New", 9))
        algo.pack(side=tk.LEFT, padx=(0, 15))
        algo.bind("<<ComboboxSelected>>", lambda e: setattr(self, "algorithm", self.algo_var.get()))

        tk.Label(left, text="SPEED:", font=("Courier New", 9, "bold"),
                bg=self.colors["panel"], fg=self.colors["muted"]).pack(side=tk.LEFT, padx=(0, 5))

        self.speed_slider = tk.Scale(left, from_=200, to=5, orient=tk.HORIZONTAL, 
                                    length=100, bg=self.colors["panel"], 
                                    fg=self.colors["text"], showvalue=0,
                                    highlightthickness=0,
                                    command=lambda v: setattr(self, "speed", int(v)))
        self.speed_slider.set(30)
        self.speed_slider.pack(side=tk.LEFT)

        right = tk.Frame(box, bg=self.colors["panel"])
        right.pack(side=tk.RIGHT)

        for text, color, cmd in [
            ("RUN", self.colors["accent"], self.run_algorithm),
            ("COMPARE", self.colors["success"], self.start_compare),
            ("CLEAR", self.colors["danger"], self.clear_walls),
            ("RESET", self.colors["muted"], self.reset_all),
        ]:
            btn = tk.Button(right, text=text, font=("Courier New", 9, "bold"),
                           bg=color, fg="white", padx=12, pady=4,
                           relief=tk.FLAT, cursor="hand2",
                           command=cmd)
            btn.pack(side=tk.LEFT, padx=2)

    def _build_legend(self, parent):
        box = tk.Frame(parent, bg=self.colors["bg"])
        box.pack(fill=tk.X, pady=(0, 5))

        modes = [
            ("start", "START", self.colors["start"]),
            ("goal", "GOAL", self.colors["goal"]),
            ("wall", "WALL", self.colors["wall"]),
            ("erase", "ERASE", self.colors["muted"]),
        ]

        self.mode_var = tk.StringVar(value="start")
        for value, label, color in modes:
            container = tk.Frame(box, bg=self.colors["bg"])
            container.pack(side=tk.LEFT, padx=(0, 15))

            swatch = tk.Label(container, bg=color, width=2, height=1,
                             relief=tk.SOLID, bd=1)
            swatch.pack(side=tk.LEFT, padx=(0, 5))

            rb = tk.Radiobutton(container, text=label, variable=self.mode_var, 
                               value=value, font=("Courier New", 8),
                               bg=self.colors["bg"], fg=self.colors["text"],
                               selectcolor=self.colors["bg"],
                               activebackground=self.colors["bg"],
                               command=lambda v=value: setattr(self, "mode", v))
            rb.pack(side=tk.LEFT)

        for label, color in [("FRONTIER", self.colors["frontier"]),
                            ("EXPLORED", self.colors["explored"]),
                            ("PATH", self.colors["path"])]:
            container = tk.Frame(box, bg=self.colors["bg"])
            container.pack(side=tk.LEFT, padx=(0, 15))

            swatch = tk.Label(container, bg=color, width=2, height=1,
                             relief=tk.SOLID, bd=1)
            swatch.pack(side=tk.LEFT, padx=(0, 5))

            tk.Label(container, text=label, font=("Courier New", 8),
                    bg=self.colors["bg"], fg=self.colors["muted"]).pack(side=tk.LEFT)

    def _cell_color(self, value):
        return {
            self.WALL: self.colors["wall"],
            self.START: self.colors["start"],
            self.GOAL: self.colors["goal"],
            self.FRONTIER: self.colors["frontier"],
            self.EXPLORED: self.colors["explored"],
            self.PATH: self.colors["path"],
        }.get(value, self.colors["empty"])

    def _redraw(self):
        self.canvas.delete("all")
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self._draw_cell(r, c)

    def _draw_cell(self, r, c):
        x1, y1 = c * self.CELL, r * self.CELL
        x2, y2 = x1 + self.CELL, y1 + self.CELL
        self.canvas.create_rectangle(x1, y1, x2, y2, 
                                    fill=self._cell_color(self.grid[r][c]),
                                    outline="#333333", width=0.5)

    def _clear_search_cells(self):
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] in (self.FRONTIER, self.EXPLORED, self.PATH):
                    self.grid[r][c] = self.EMPTY
        self._redraw()

    def _cell_at(self, event):
        c, r = event.x // self.CELL, event.y // self.CELL
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            return r, c
        return None

    def on_click(self, event):
        if self.is_running or self.comparing:
            return
        cell = self._cell_at(event)
        if cell:
            self._apply_mode(*cell)

    def on_drag(self, event):
        if self.is_running or self.comparing or self.mode not in ("wall", "erase"):
            return
        cell = self._cell_at(event)
        if cell:
            self._apply_mode(*cell)

    def on_erase_wall(self, event):
        if self.is_running or self.comparing:
            return
        cell = self._cell_at(event)
        if cell and self.grid[cell[0]][cell[1]] == self.WALL:
            self.grid[cell[0]][cell[1]] = self.EMPTY
            self._draw_cell(*cell)

    def _apply_mode(self, r, c):
        if self.mode == "start":
            if self.start:
                self.grid[self.start[0]][self.start[1]] = self.EMPTY
                self._draw_cell(*self.start)
            self.start = (r, c)
            self.grid[r][c] = self.START
            self._draw_cell(r, c)
            self.status.set("[ READY ] Place goal node")

        elif self.mode == "goal":
            if self.goal:
                self.grid[self.goal[0]][self.goal[1]] = self.EMPTY
                self._draw_cell(*self.goal)
            self.goal = (r, c)
            self.grid[r][c] = self.GOAL
            self._draw_cell(r, c)
            self.status.set("[ READY ] Press RUN")

        elif self.mode == "wall":
            if self.grid[r][c] == self.EMPTY and (r, c) not in (self.start, self.goal):
                self.grid[r][c] = self.WALL
                self._draw_cell(r, c)

        elif self.mode == "erase":
            if self.grid[r][c] == self.WALL:
                self.grid[r][c] = self.EMPTY
                self._draw_cell(r, c)

    def run_algorithm(self):
        if self.is_running or self.comparing or not self._ready_to_run():
            return
        self._clear_search_cells()
        self.is_running = True
        self.status.set(f"[ RUNNING ] {self.algorithm} ...")

        mode = ALGORITHMS[self.algorithm]
        generator = run_search(self.grid, self.start, self.goal, mode)
        self._step(generator, time.time())

    def _ready_to_run(self):
        if self.start is None:
            messagebox.showinfo("Error", "Place a start node first.")
            return False
        if self.goal is None:
            messagebox.showinfo("Error", "Place a goal node first.")
            return False
        if self.start == self.goal:
            messagebox.showinfo("Error", "Start and goal can't be the same.")
            return False
        return True

    def _step(self, generator, start_time):
        try:
            frontier, explored, current, path = next(generator)
            self._paint_progress(frontier, explored, current, path)
            self.anim_job = self.root.after(self.speed, self._step, generator, start_time)
        except StopIteration:
            self._finish_run(start_time)

    def _paint_progress(self, frontier, explored, current, path):
        self._clear_search_cells()

        for r, c in frontier:
            if self.grid[r][c] == self.EMPTY:
                self.grid[r][c] = self.FRONTIER
        for r, c in explored:
            if self.grid[r][c] in (self.EMPTY, self.FRONTIER):
                self.grid[r][c] = self.EXPLORED
        if current and self.grid[current[0]][current[1]] not in (self.START, self.GOAL):
            self.grid[current[0]][current[1]] = self.FRONTIER
        if path:
            for r, c in path:
                if self.grid[r][c] not in (self.START, self.GOAL):
                    self.grid[r][c] = self.PATH
        self._redraw()

    def _finish_run(self, start_time):
        elapsed = round((time.time() - start_time) * 1000, 2)
        path = [(r, c) for r in range(self.ROWS) for c in range(self.COLS) 
                if self.grid[r][c] == self.PATH]
        expanded = sum(1 for r in range(self.ROWS) for c in range(self.COLS) 
                      if self.grid[r][c] == self.EXPLORED)

        self.stats.append((self.algorithm, len(path), expanded, elapsed))
        self._refresh_results()

        if path:
            self.status.set(f"[ DONE ] {self.algorithm} - Path: {len(path)} cells")
        else:
            self.status.set(f"[ FAIL ] {self.algorithm} - No path found")

        self.is_running = False
        self._highlight_best()

    def start_compare(self):
        if self.is_running or self.comparing or not self._ready_to_run():
            return
        
        self.comparing = True
        self.stats = []
        self._refresh_results()
        self._clear_search_cells()
        
        self.status.set("[ COMPARING ] Running all algorithms...")
        self.base_grid = [row[:] for row in self.grid]
        self.base_start = self.start
        self.base_goal = self.goal
        self.compare_index = 0
        self.compare_algorithms = list(ALGORITHMS.items())
        
        self._run_next_compare()

    def _run_next_compare(self):
        if self.compare_index >= len(self.compare_algorithms):
            self._finish_compare()
            return
        
        name, mode = self.compare_algorithms[self.compare_index]
        self.status.set(f"[ COMPARING ] {name} ({self.compare_index + 1}/{len(self.compare_algorithms)})...")
        
        test_grid = [row[:] for row in self.base_grid]
        self.grid = test_grid
        self._clear_search_cells()
        
        generator = run_search(test_grid, self.base_start, self.base_goal, mode)
        self.compare_generator = generator
        self.compare_name = name
        self.compare_start_time = time.time()
        
        self._compare_step()

    def _compare_step(self):
        try:
            frontier, explored, current, path = next(self.compare_generator)
            self._paint_progress(frontier, explored, current, path)
            self.anim_job = self.root.after(self.speed, self._compare_step)
        except StopIteration:
            self._finish_compare_algorithm()

    def _finish_compare_algorithm(self):
        elapsed = round((time.time() - self.compare_start_time) * 1000, 2)
        path = [(r, c) for r in range(self.ROWS) for c in range(self.COLS) 
                if self.grid[r][c] == self.PATH]
        expanded = sum(1 for r in range(self.ROWS) for c in range(self.COLS) 
                      if self.grid[r][c] == self.EXPLORED)
        
        self.stats.append((self.compare_name, len(path), expanded, elapsed))
        self._refresh_results()
        self._highlight_best()
        
        if path:
            self.status.set(f"[ DONE ] {self.compare_name} - Path: {len(path)} cells")
        else:
            self.status.set(f"[ DONE ] {self.compare_name} - No path found")
        
        self.compare_index += 1
        self.root.after(400, self._run_next_compare)

    def _finish_compare(self):
        self.comparing = False
        self.compare_generator = None
        self.status.set("[ COMPLETE ] Comparison finished")
        
        if self.stats:
            best = min([s for s in self.stats if s[1] > 0], 
                      key=lambda x: (x[1], x[2]), default=None)
            if best:
                self.status.set(f"[ BEST ] {best[0]} - Path: {best[1]} cells")

    def _refresh_results(self):
        self.tree.delete(*self.tree.get_children())
        for row in self.stats:
            self.tree.insert("", tk.END, values=row)

    def _highlight_best(self):
        candidates = [row for row in self.stats if row[1] > 0]
        if not candidates:
            self.best_label.config(text="No algorithm found a path", 
                                  fg=self.colors["danger"])
            return

        best = min(candidates, key=lambda row: (row[1], row[2]))
        self.best_label.config(
            text=f"BEST: {best[0]} - Path: {best[1]} cells, Expanded: {best[2]}, Time: {best[3]}ms",
            fg=self.colors["success"]
        )

    def _build_results(self, parent):
        frame = tk.Frame(parent, bg=self.colors["bg"])
        frame.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(frame, bg=self.colors["bg"])
        header.pack(fill=tk.X, pady=(0, 5))

        tk.Label(header, text="RESULTS", font=("Courier New", 11, "bold"),
                bg=self.colors["bg"], fg=self.colors["header"]).pack(side=tk.LEFT)

        clear_btn = tk.Button(header, text="CLEAR", font=("Courier New", 8),
                             bg=self.colors["panel"], fg=self.colors["text"],
                             padx=8, pady=2, relief=tk.FLAT,
                             cursor="hand2", command=self.clear_stats)
        clear_btn.pack(side=tk.RIGHT)

        columns = ("Algorithm", "Path", "Expanded", "Time (ms)")
        
        tree_box = tk.Frame(frame, bg=self.colors["border"], padx=1, pady=1)
        tree_box.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Treeview", 
                       font=("Courier New", 9),
                       background=self.colors["panel"],
                       fieldbackground=self.colors["panel"],
                       foreground=self.colors["text"])
        style.configure("Custom.Treeview.Heading", 
                       font=("Courier New", 9, "bold"),
                       background=self.colors["panel"],
                       foreground=self.colors["text"])
        
        self.tree = ttk.Treeview(tree_box, columns=columns, show="headings",
                                height=6, style="Custom.Treeview")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.best_label = tk.Label(frame, text="", font=("Courier New", 10, "bold"),
                                   bg=self.colors["bg"], fg=self.colors["success"])
        self.best_label.pack(anchor="w", pady=(5, 0))

    def clear_stats(self):
        self.stats = []
        self._refresh_results()
        self.best_label.config(text="")

    def clear_walls(self):
        if self.is_running or self.comparing:
            return
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == self.WALL:
                    self.grid[r][c] = self.EMPTY
        self._redraw()

    def reset_all(self):
        if self.is_running:
            return
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None

        self.comparing = False
        self.compare_generator = None
        self.compare_index = 0
        self.grid = [[self.EMPTY] * self.COLS for _ in range(self.ROWS)]
        self.start = None
        self.goal = None
        self.clear_stats()
        self.status.set("[ READY ] Place start node")
        self._redraw()

def main():
    root = tk.Tk()
    app = SearchVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()