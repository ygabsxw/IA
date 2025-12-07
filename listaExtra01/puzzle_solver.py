import time
import heapq
import math
from collections import deque
import os

# --- Configurações do Jogo ---
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
MOVES = {
    'Cima': -3, 'Baixo': 3, 'Esquerda': -1, 'Direita': 1
}

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost  # g(n)
        self.heuristic = heuristic  # h(n)
        self.total_score = cost + heuristic  # f(n)

    # Necessário para a fila de prioridade comparar nós
    def __lt__(self, other):
        return self.total_score < other.total_score

    def get_path(self):
        path = []
        node = self
        while node:
            path.append((node.action, node.state))
            node = node.parent
        return path[::-1]

def get_blank_pos(state):
    return state.index(0)

def get_neighbors(state):
    neighbors = []
    blank = get_blank_pos(state)
    row, col = divmod(blank, 3)

    possible_moves = []
    if row > 0: possible_moves.append(('Cima', -3))
    if row < 2: possible_moves.append(('Baixo', 3))
    if col > 0: possible_moves.append(('Esquerda', -1))
    if col < 2: possible_moves.append(('Direita', 1))

    for action, move in possible_moves:
        new_blank = blank + move
        new_state = list(state)
        # Swap
        new_state[blank], new_state[new_blank] = new_state[new_blank], new_state[blank]
        neighbors.append((tuple(new_state), action))
    return neighbors

def h_misplaced(state):
    """Conta quantas peças não estão na posição correta."""
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL_STATE[i])

def h_manhattan(state):
    """Soma das distâncias absolutas (x, y) de cada peça até seu alvo."""
    distance = 0
    for i, tile in enumerate(state):
        if tile == 0: continue
        target_idx = GOAL_STATE.index(tile)
        x1, y1 = divmod(i, 3)
        x2, y2 = divmod(target_idx, 3)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def h_euclidean(state):
    """Soma das distâncias euclidianas (linha reta) de cada peça até seu alvo."""
    distance = 0
    for i, tile in enumerate(state):
        if tile == 0: continue
        target_idx = GOAL_STATE.index(tile)
        x1, y1 = divmod(i, 3)
        x2, y2 = divmod(target_idx, 3)
        distance += math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

# --- Algoritmos de Busca ---

def solve_bfs(start_state):
    """Busca em Largura"""
    start_node = Node(start_state)
    if start_state == GOAL_STATE: return start_node, 0

    frontier = deque([start_node])
    explored = {start_state}
    nodes_visited = 0

    while frontier:
        node = frontier.popleft()
        nodes_visited += 1

        for neighbor_state, action in get_neighbors(node.state):
            if neighbor_state not in explored:
                child = Node(neighbor_state, node, action, node.cost + 1)
                if neighbor_state == GOAL_STATE:
                    return child, nodes_visited
                explored.add(neighbor_state)
                frontier.append(child)
    return None, nodes_visited

def solve_greedy(start_state, heuristic_func=h_manhattan):
    """Busca Gulosa (usa apenas h(n))"""
    start_node = Node(start_state, heuristic=heuristic_func(start_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = {start_state}
    nodes_visited = 0

    while frontier:
        node = heapq.heappop(frontier)
        nodes_visited += 1

        if node.state == GOAL_STATE:
            return node, nodes_visited

        for neighbor_state, action in get_neighbors(node.state):
            if neighbor_state not in explored:
                h = heuristic_func(neighbor_state)
                # Na gulosa, o custo g(n) não importa para a ordenação, apenas h(n)
                # Mas mantemos cost calculado para saber o tamanho do caminho final
                child = Node(neighbor_state, node, action, cost=node.cost + 1, heuristic=h)
                # Forçamos o score total ser apenas h para a priority queue
                child.total_score = h 
                explored.add(neighbor_state)
                heapq.heappush(frontier, child)
    return None, nodes_visited

def solve_astar(start_state, heuristic_func):
    """Algoritmo A* (usa g(n) + h(n))"""
    start_node = Node(start_state, cost=0, heuristic=heuristic_func(start_state))
    frontier = []
    heapq.heappush(frontier, start_node)
    explored = {} # Mantém o menor custo encontrado para um estado
    explored[start_state] = 0
    nodes_visited = 0

    while frontier:
        node = heapq.heappop(frontier)
        nodes_visited += 1

        if node.state == GOAL_STATE:
            return node, nodes_visited
        
        # Se encontramos um caminho melhor para este nó antes de processá-lo, pula
        if node.cost > explored[node.state]:
            continue

        for neighbor_state, action in get_neighbors(node.state):
            new_cost = node.cost + 1
            if neighbor_state not in explored or new_cost < explored[neighbor_state]:
                h = heuristic_func(neighbor_state)
                child = Node(neighbor_state, node, action, cost=new_cost, heuristic=h)
                explored[neighbor_state] = new_cost
                heapq.heappush(frontier, child)
    return None, nodes_visited

def print_board(state):
    print(f"{state[0]} | {state[1]} | {state[2]}")
    print(f"{state[3]} | {state[4]} | {state[5]}")
    print(f"{state[6]} | {state[7]} | {state[8]}")
    print("-" * 9)

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=== RESOLVEDOR DO JOGO DOS 8 ===")
    print("Digite o estado inicial (9 números de 0 a 8, separados por espaço):")
    
    try:
        user_input = input("Estado: ").strip().split()
        if len(user_input) != 9:
            raise ValueError
        start_state = tuple(map(int, user_input))
    except:
        print("Entrada inválida. Usando exemplo padrão.")
        start_state = (1, 2, 3, 4, 5, 6, 0, 7, 8)

    print(f"\nEstado Inicial Definido: {start_state}")
    print_board(start_state)

    while True:
        print("\nEscolha o Algoritmo:")
        print("1. Busca em Largura (BFS)")
        print("2. Busca Gulosa (Greedy) - Heurística Manhattan")
        print("3. A* - Heurística: Peças Fora do Lugar")
        print("4. A* - Heurística: Distância de Manhattan")
        print("5. A* - Heurística: Distância Euclidiana")
        print("0. Sair")
        
        choice = input("Opção: ")
        
        if choice == '0': break
        
        algorithm_name = ""
        result_node = None
        visited = 0
        start_time = time.time()
        
        if choice == '1':
            algorithm_name = "BFS"
            result_node, visited = solve_bfs(start_state)
        elif choice == '2':
            algorithm_name = "Greedy (Manhattan)"
            result_node, visited = solve_greedy(start_state, h_manhattan)
        elif choice == '3':
            algorithm_name = "A* (Misplaced Tiles)"
            result_node, visited = solve_astar(start_state, h_misplaced)
        elif choice == '4':
            algorithm_name = "A* (Manhattan)"
            result_node, visited = solve_astar(start_state, h_manhattan)
        elif choice == '5':
            algorithm_name = "A* (Euclidean)"
            result_node, visited = solve_astar(start_state, h_euclidean)
        else:
            print("Opção inválida.")
            continue

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n--- Resultados para {algorithm_name} ---")
        if result_node:
            path = result_node.get_path()
            print(f"Tempo: {duration:.4f} segundos")
            print(f"Nós visitados: {visited}")
            print(f"Profundidade da solução (passos): {len(path) - 1}")
            
            print("\nCaminho Encontrado:")
            for i, (action, state) in enumerate(path):
                if action:
                    print(f"Passo {i}: {action}")
                print_board(state)
        else:
            print("Não foi encontrada solução (ou tempo excedido/memória cheia).")
        
        input("Pressione Enter para continuar...")

if __name__ == "__main__":
    main()