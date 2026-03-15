#undef _GLIBCXX_DEBUG
#pragma GCC optimize("O3,inline,unroll-loops,omit-frame-pointer")

#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace std::chrono;
typedef steady_clock hrclock;

// ─── Constants ───────────────────────────────────────────
constexpr int MAX_WIDTH         = 50;
constexpr int MAX_HEIGHT        = 50;
constexpr int MAX_CELLS         = MAX_WIDTH * MAX_HEIGHT;
constexpr int MAX_SNAKES        = 8;
constexpr int MAX_PARTS         = 100;
constexpr int MAX_POWER_SOURCES = 100;

enum Direction : int8_t { UP=0, DOWN=1, LEFT=2, RIGHT=3, NONE=4 };
constexpr int DIR_X[] = {0, 0, -1, 1, 0};
constexpr int DIR_Y[] = {-1, 1, 0, 0, 0};
const char* DIR_NAMES[] = {"UP","DOWN","LEFT","RIGHT","WAIT"};

int  g_width, g_height, g_total;
bool g_wall[MAX_CELLS];

// ─── Tunable Parameters ─────────────────────────────────
#ifndef PARAM_DEPTH
#define PARAM_DEPTH 6
#endif
#ifndef PARAM_PUCT
#define PARAM_PUCT 0.1
#endif

constexpr int    DUCT_MAX_DEPTH = PARAM_DEPTH;
constexpr double PUCT_C         = PARAM_PUCT;

// ─── Data Structures ────────────────────────────────────
struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x==o.x && y==o.y; }
};
struct Move { Direction dir; };
struct MoveList {
    Move moves[5];
    int count = 0;
    void add(Direction d) { moves[count++] = {d}; }
};
struct Snakebot {
    int id;
    bool is_alive;
    bool is_mine;
    int part_count;
    Point parts[MAX_PARTS];
};
struct GameState {
    int turn;
    int power_source_count;
    Point power_sources[MAX_POWER_SOURCES];
    int snake_count;
    Snakebot snakes[MAX_SNAKES];
};

// ─── Timing ─────────────────────────────────────────────
inline int get_time_us(time_point<hrclock> start) {
    return (int)duration_cast<microseconds>(hrclock::now() - start).count();
}

// ─── State Copy ─────────────────────────────────────────
inline void fast_copy_state(GameState& dst, const GameState& src) {
    dst.turn = src.turn;
    dst.power_source_count = src.power_source_count;
    memcpy(dst.power_sources, src.power_sources,
           src.power_source_count * sizeof(Point));
    dst.snake_count = src.snake_count;
    for (int i = 0; i < src.snake_count; i++) {
        dst.snakes[i].id         = src.snakes[i].id;
        dst.snakes[i].is_alive   = src.snakes[i].is_alive;
        dst.snakes[i].is_mine    = src.snakes[i].is_mine;
        dst.snakes[i].part_count = src.snakes[i].part_count;
        if (src.snakes[i].is_alive)
            memcpy(dst.snakes[i].parts, src.snakes[i].parts,
                   src.snakes[i].part_count * sizeof(Point));
    }
}

// ─── Grid-based Occupancy ───────────────────────────────
int g_occ_val[MAX_CELLS];
int g_occ_gen[MAX_CELLS];
int g_occ_g = 0;

inline void occ_clear() { g_occ_g++; }
inline void occ_set(int x, int y, int snake_idx) {
    if ((unsigned)x >= (unsigned)g_width || (unsigned)y >= (unsigned)g_height) return;
    int idx = y * g_width + x;
    g_occ_gen[idx] = g_occ_g;
    g_occ_val[idx] = snake_idx;
}
inline bool occ_has(int x, int y) {
    if ((unsigned)x >= (unsigned)g_width || (unsigned)y >= (unsigned)g_height) return false;
    return g_occ_gen[y * g_width + x] == g_occ_g;
}
inline int occ_get(int x, int y) {
    if ((unsigned)x >= (unsigned)g_width || (unsigned)y >= (unsigned)g_height) return -1;
    int idx = y * g_width + x;
    return (g_occ_gen[idx] == g_occ_g) ? g_occ_val[idx] : -1;
}

int g_food_gen[MAX_CELLS];
int g_food_g = 0;
inline void food_grid_build(const GameState& state) {
    g_food_g++;
    for (int f = 0; f < state.power_source_count; f++) {
        int x = state.power_sources[f].x, y = state.power_sources[f].y;
        if ((unsigned)x < (unsigned)g_width && (unsigned)y < (unsigned)g_height)
            g_food_gen[y * g_width + x] = g_food_g;
    }
}
inline bool food_at(int x, int y) {
    if ((unsigned)x >= (unsigned)g_width || (unsigned)y >= (unsigned)g_height) return false;
    return g_food_gen[y * g_width + x] == g_food_g;
}

void build_occupancy(const GameState& state) {
    occ_clear();
    for (int s = 0; s < state.snake_count; s++) {
        if (!state.snakes[s].is_alive) continue;
        int end = state.snakes[s].part_count - 1;
        for (int p = 0; p < end; p++)
            occ_set(state.snakes[s].parts[p].x, state.snakes[s].parts[p].y, s);
    }
}

void build_full_occupancy(const GameState& state) {
    occ_clear();
    for (int s = 0; s < state.snake_count; s++) {
        if (!state.snakes[s].is_alive) continue;
        for (int p = 0; p < state.snakes[s].part_count; p++)
            occ_set(state.snakes[s].parts[p].x, state.snakes[s].parts[p].y, s);
    }
}

inline bool is_blocked_cell(int x, int y) {
    int idx = y * g_width + x;
    return g_wall[idx] || g_occ_gen[idx] == g_occ_g;
}

// ─── Parse Body ─────────────────────────────────────────
void parse_body(const string& body_str, Snakebot& bot) {
    bot.part_count = 0;
    int i = 0, len = body_str.length();
    while (i < len) {
        int x=0, y=0;
        while (i < len && body_str[i] != ',') { x = x*10 + (body_str[i]-'0'); i++; }
        i++;
        while (i < len && body_str[i] != ':') { y = y*10 + (body_str[i]-'0'); i++; }
        i++;
        bot.parts[bot.part_count++] = {x, y};
    }
}

// ─── Move Generation ────────────────────────────────────
MoveList generate_moves(const Snakebot& bot) {
    MoveList list, fallback;
    if (!bot.is_alive) { list.add(NONE); return list; }
    for (int d = 0; d < 4; d++) {
        int nx = bot.parts[0].x + DIR_X[d];
        int ny = bot.parts[0].y + DIR_Y[d];
        if (bot.part_count > 1 && nx == bot.parts[1].x && ny == bot.parts[1].y)
            continue;
        fallback.add((Direction)d);
        if ((unsigned)nx >= (unsigned)g_width ||
            (unsigned)ny >= (unsigned)g_height) continue;
        if (is_blocked_cell(nx, ny)) continue;
        list.add((Direction)d);
    }
    if (list.count == 0 && fallback.count == 0) fallback.add(UP); // Desperation
    return list.count == 0 ? fallback : list;
}

// ─── Simulation (with gravity) ──────────────────────────
inline void occ_remove_snake(int si, const Snakebot& s) {
    for (int p = 0; p < s.part_count; p++) {
        int x = s.parts[p].x, y = s.parts[p].y;
        if ((unsigned)x < (unsigned)g_width && (unsigned)y < (unsigned)g_height) {
            int idx = y * g_width + x;
            if (g_occ_gen[idx] == g_occ_g && g_occ_val[idx] == si)
                g_occ_gen[idx] = 0;
        }
    }
}

inline void occ_add_snake(int si, const Snakebot& s) {
    for (int p = 0; p < s.part_count; p++)
        occ_set(s.parts[p].x, s.parts[p].y, si);
}

void simulate(GameState& state, const Move* moves) {
    Point new_heads[MAX_SNAKES];
    bool grew[MAX_SNAKES] = {};
    for (int i = 0; i < state.snake_count; i++) {
        if (!state.snakes[i].is_alive || moves[i].dir == NONE) {
            new_heads[i] = state.snakes[i].parts[0];
            continue;
        }
        new_heads[i].x = state.snakes[i].parts[0].x + DIR_X[moves[i].dir];
        new_heads[i].y = state.snakes[i].parts[0].y + DIR_Y[moves[i].dir];
    }

    for (int p = 0; p < state.power_source_count; p++) {
        bool eaten = false;
        for (int i = 0; i < state.snake_count; i++) {
            if (state.snakes[i].is_alive && moves[i].dir != NONE && new_heads[i] == state.power_sources[p]) {
                grew[i] = true; eaten = true;
            }
        }
        if (eaten) {
            state.power_sources[p] = state.power_sources[--state.power_source_count];
            p--;
        }
    }

    for (int i = 0; i < state.snake_count; i++) {
        if (!state.snakes[i].is_alive || moves[i].dir == NONE) continue;
        if (grew[i]) {
            if (state.snakes[i].part_count >= MAX_PARTS) grew[i] = false;
            else state.snakes[i].part_count++;
        }
        if (state.snakes[i].part_count > 1)
            memmove(&state.snakes[i].parts[1], &state.snakes[i].parts[0],
                    (state.snakes[i].part_count - 1) * sizeof(Point));
        state.snakes[i].parts[0] = new_heads[i];
    }

    build_full_occupancy(state);
    bool head_destroyed[MAX_SNAKES] = {};
    for (int i = 0; i < state.snake_count; i++) {
        if (!state.snakes[i].is_alive || moves[i].dir == NONE) continue;
        Point head = state.snakes[i].parts[0];
        if ((unsigned)head.x >= (unsigned)g_width ||
            (unsigned)head.y >= (unsigned)g_height) continue;
        bool hit = false;
        if (g_wall[head.y * g_width + head.x]) hit = true;
        if (!hit) {
            int occ_snake = occ_get(head.x, head.y);
            if (occ_snake >= 0 && occ_snake != i) {
                hit = true;
            } else if (occ_snake == i) {
                for (int p = 1; p < state.snakes[i].part_count; p++) {
                    if (head == state.snakes[i].parts[p]) { hit = true; break; }
                }
            }
        }
        if (!hit) {
            for (int j = 0; j < state.snake_count; j++) {
                if (j == i || !state.snakes[j].is_alive) continue;
                if (head == state.snakes[j].parts[0]) { hit = true; break; }
            }
        }
        if (hit) head_destroyed[i] = true;
    }
    for (int i = 0; i < state.snake_count; i++) {
        if (!state.snakes[i].is_alive || !head_destroyed[i]) continue;
        if (state.snakes[i].part_count <= 3) {
            state.snakes[i].is_alive = false;
        } else {
            memmove(&state.snakes[i].parts[0], &state.snakes[i].parts[1],
                    (state.snakes[i].part_count - 1) * sizeof(Point));
            state.snakes[i].part_count--;
        }
    }

    food_grid_build(state);
    build_full_occupancy(state);

    {
        bool all_wall_grounded = true;
        for (int i = 0; i < state.snake_count && all_wall_grounded; i++) {
            if (!state.snakes[i].is_alive) continue;
            bool has_wall = false;
            for (int p = 0; p < state.snakes[i].part_count; p++) {
                int bx = state.snakes[i].parts[p].x;
                int by = state.snakes[i].parts[p].y + 1;
                if ((unsigned)bx < (unsigned)g_width && (unsigned)by < (unsigned)g_height &&
                    g_wall[by * g_width + bx]) {
                    has_wall = true; break;
                }
            }
            if (!has_wall) all_wall_grounded = false;
        }
        if (all_wall_grounded) return;
    }

    bool something_fell_outer = true;
    int gravity_iters = 0;
    while (something_fell_outer && ++gravity_iters <= g_height + 10) {
        bool individual_fell = true;
        while (individual_fell) {
            individual_fell = false;
            for (int i = 0; i < state.snake_count; i++) {
                if (!state.snakes[i].is_alive) continue;
                bool supported = false;
                for (int p = 0; p < state.snakes[i].part_count; p++) {
                    int bx = state.snakes[i].parts[p].x;
                    int by = state.snakes[i].parts[p].y + 1;
                    if ((unsigned)bx >= (unsigned)g_width || (unsigned)by >= (unsigned)g_height)
                        continue;
                    int bidx = by * g_width + bx;
                    if (g_wall[bidx]) { supported = true; break; }
                    if (g_food_gen[bidx] == g_food_g) { supported = true; break; }
                    if (g_occ_gen[bidx] == g_occ_g && g_occ_val[bidx] != i) {
                        supported = true; break;
                    }
                }
                if (!supported) {
                    individual_fell = true;
                    occ_remove_snake(i, state.snakes[i]);
                    for (int p = 0; p < state.snakes[i].part_count; p++)
                        state.snakes[i].parts[p].y++;
                    bool all_off = true;
                    for (int p = 0; p < state.snakes[i].part_count; p++) {
                        if (state.snakes[i].parts[p].y < g_height + 1) { all_off = false; break; }
                    }
                    if (all_off) {
                        state.snakes[i].is_alive = false;
                    } else {
                        occ_add_snake(i, state.snakes[i]);
                    }
                }
            }
        }

        something_fell_outer = false;
        build_full_occupancy(state);

        bool visited[MAX_SNAKES] = {};
        for (int i = 0; i < state.snake_count; i++) {
            if (visited[i] || !state.snakes[i].is_alive) continue;
            int group[MAX_SNAKES], gsize = 0;
            int stk[MAX_SNAKES], stop = 0;
            stk[stop++] = i; visited[i] = true;
            while (stop > 0) {
                int cur = stk[--stop];
                group[gsize++] = cur;
                for (int p = 0; p < state.snakes[cur].part_count; p++) {
                    int px = state.snakes[cur].parts[p].x;
                    int py = state.snakes[cur].parts[p].y;
                    for (int d = 0; d < 4; d++) {
                        int neighbor = occ_get(px + DIR_X[d], py + DIR_Y[d]);
                        if (neighbor >= 0 && neighbor != cur && !visited[neighbor] &&
                            state.snakes[neighbor].is_alive) {
                            visited[neighbor] = true;
                            stk[stop++] = neighbor;
                        }
                    }
                }
            }
            if (gsize <= 1) continue;
            bool group_supported = false;
            bool in_group[MAX_SNAKES] = {};
            for (int g = 0; g < gsize; g++) in_group[group[g]] = true;
            for (int g = 0; g < gsize && !group_supported; g++) {
                int si = group[g];
                for (int p = 0; p < state.snakes[si].part_count && !group_supported; p++) {
                    int bx = state.snakes[si].parts[p].x;
                    int by = state.snakes[si].parts[p].y + 1;
                    if ((unsigned)bx >= (unsigned)g_width || (unsigned)by >= (unsigned)g_height) continue;
                    int bidx = by * g_width + bx;
                    if (g_wall[bidx]) { group_supported = true; break; }
                    if (g_food_gen[bidx] == g_food_g) { group_supported = true; break; }
                    int occ_s = (g_occ_gen[bidx] == g_occ_g) ? g_occ_val[bidx] : -1;
                    if (occ_s >= 0 && !in_group[occ_s]) { group_supported = true; break; }
                }
            }
            if (group_supported) continue;
            something_fell_outer = true;
            for (int g = 0; g < gsize; g++) {
                int si = group[g];
                occ_remove_snake(si, state.snakes[si]);
                for (int p = 0; p < state.snakes[si].part_count; p++)
                    state.snakes[si].parts[p].y++;
                if (state.snakes[si].parts[0].y >= g_height)
                    state.snakes[si].is_alive = false;
                else
                    occ_add_snake(si, state.snakes[si]);
            }
        }
    }
}

// ─── Evaluation ─────────────────────────────────────────
double evaluate(const GameState& state) {
    bool my_alive = false, opp_alive = false;
    int my_length = 0, opp_length = 0;
    int my_count = 0, opp_count = 0;
    int my_grounded = 0, opp_grounded = 0;

    static int bfs_vis[MAX_CELLS] = {0};
    static int bfs_gen = 1;

    int my_space = 0, opp_space = 0;
    int dist_to_food[MAX_SNAKES];
    for(int i=0; i<MAX_SNAKES; ++i) dist_to_food[i] = 999;

    for (int i = 0; i < state.snake_count; i++) {
        const Snakebot& s = state.snakes[i];
        if (!s.is_alive) continue;
        bool grounded = false;
        for (int p = 0; p < s.part_count && !grounded; p++) {
            int bx = s.parts[p].x, by = s.parts[p].y + 1;
            if ((unsigned)bx < (unsigned)g_width && (unsigned)by < (unsigned)g_height)
                if (g_wall[by * g_width + bx]) grounded = true;
        }

        bfs_gen++;
        int head_x = s.parts[0].x, head_y = s.parts[0].y;
        int q[100], dists[100];
        int head_ptr = 0, tail_ptr = 0;
        int head_idx = head_y * g_width + head_x;
        q[tail_ptr] = head_idx;
        dists[tail_ptr++] = 0;
        bfs_vis[head_idx] = bfs_gen;

        int space = 0;
        int closest_food = 999;

        while (head_ptr < tail_ptr && tail_ptr < 60) {
            int cur = q[head_ptr];
            int d = dists[head_ptr++];
            space++;
            if (state.power_source_count > 0 && closest_food == 999) {
                if (g_food_gen[cur] == g_food_g) closest_food = d;
            }
            int cx = cur % g_width;
            int cy = cur / g_width;
            for (int dir=0; dir<4; dir++) {
                int nx = cx + DIR_X[dir];
                int ny = cy + DIR_Y[dir];
                if ((unsigned)nx < (unsigned)g_width && (unsigned)ny < (unsigned)g_height) {
                    int nidx = ny * g_width + nx;
                    if (g_wall[nidx]) continue;
                    int occ = (g_occ_gen[nidx] == g_occ_g) ? g_occ_val[nidx] : -1;
                    if (occ != -1 && occ != i) continue;
                    if (bfs_vis[nidx] != bfs_gen) {
                        bfs_vis[nidx] = bfs_gen;
                        q[tail_ptr] = nidx;
                        dists[tail_ptr] = d + 1;
                        tail_ptr++;
                        if (tail_ptr >= 60) break;
                    }
                }
            }
        }
        dist_to_food[i] = closest_food;

        if (s.is_mine) {
            my_alive = true; my_length += s.part_count; my_count++;
            if (grounded) my_grounded++;
            my_space += space;
        } else {
            opp_alive = true; opp_length += s.part_count; opp_count++;
            if (grounded) opp_grounded++;
            opp_space += space;
        }
    }

    if (!my_alive && !opp_alive) return 0.0;
    if (!my_alive) return -1.0;
    if (!opp_alive) return 1.0;

    if (state.power_source_count == 0 || (my_length > 0 && opp_length > 0 && (my_length + opp_length > 30))) {
        int diff = my_length - opp_length;
        if (diff > 0) return 0.8 + 0.15 * min(diff, 10) / 10.0;
        if (diff < 0) return -(0.8 + 0.15 * min(-diff, 10) / 10.0);
        return 0.0;
    }

    double score = 0.0;
    int total_len = max(my_length + opp_length, 1);
    
    score += 0.45 * (double)(my_length - opp_length) / total_len;
    score += 0.10 * (double)(my_count - opp_count) / max(my_count + opp_count, 1);
    score += 0.05 * (my_grounded - opp_grounded);

    int total_space = max(my_space + opp_space, 1);
    score += 0.20 * (double)(my_space - opp_space) / total_space;

    if (state.power_source_count > 0) {
        int my_best = 999, opp_best = 999;
        for (int i=0; i<state.snake_count; i++) {
            if (!state.snakes[i].is_alive) continue;
            if (state.snakes[i].is_mine) my_best = min(my_best, dist_to_food[i]);
            else opp_best = min(opp_best, dist_to_food[i]);
        }
        int sum_dist = max(my_best + opp_best, 1);
        score += 0.20 * (double)(opp_best - my_best) / sum_dist;
    }

    score = score * 1.5;
    if (score > 1.0) score = 1.0;
    if (score < -1.0) score = -1.0;
    return score;
}

// ═══════════════════════════════════════════════════════════
// TRUE DECOUPLED UCT TREE (Memory Safe)
// ═══════════════════════════════════════════════════════════
constexpr int MAX_NODES = 100000;
constexpr int CHILD_MAX = 32;

struct MCTSNode {
    int total_visits;
    int visits[MAX_SNAKES][5];
    double scores[MAX_SNAKES][5];
    
    int child_keys[CHILD_MAX];
    int child_nodes[CHILD_MAX];
    int child_count;

    void init() {
        total_visits = 0;
        memset(visits, 0, sizeof(visits));
        memset(scores, 0, sizeof(scores));
        memset(child_keys, -1, sizeof(child_keys));
        memset(child_nodes, -1, sizeof(child_nodes));
        child_count = 0;
    }
};

MCTSNode g_tree[MAX_NODES];
int g_node_count = 0;

inline int alloc_node() {
    if (g_node_count < MAX_NODES) {
        g_tree[g_node_count].init();
        return g_node_count++;
    }
    return -1;
}

// ─── Smart Priors (Gravity & Anti-Jump Aware) ───────────
inline double compute_move_prior(const Snakebot& snake, Direction dir, const GameState& state) {
    if (dir == NONE) return 1.0;
    int nx = snake.parts[0].x + DIR_X[dir];
    int ny = snake.parts[0].y + DIR_Y[dir];
    double prior = 1.0;

    if ((unsigned)nx >= (unsigned)g_width || (unsigned)ny >= (unsigned)g_height || g_wall[ny * g_width + nx]) {
        return 0.01;
    }

    // Anti-Jump (Yo-Yo) Logic
    if (dir == UP) {
        bool can_climb = false;
        int hx = snake.parts[0].x, hy = snake.parts[0].y;
        
        // Check for adjacent walls to grip
        if (nx > 0 && g_wall[ny * g_width + nx - 1]) can_climb = true;
        if (nx < g_width - 1 && g_wall[ny * g_width + nx + 1]) can_climb = true;
        if (hx > 0 && g_wall[hy * g_width + hx - 1]) can_climb = true;
        if (hx < g_width - 1 && g_wall[hy * g_width + hx + 1]) can_climb = true;

        // Check if body is anchored
        bool anchored = false;
        for (int p = 0; p < snake.part_count; p++) {
            int px = snake.parts[p].x, py = snake.parts[p].y;
            if (py + 1 < g_height && g_wall[(py + 1) * g_width + px]) {
                anchored = true; break;
            }
        }
        
        if (!can_climb && !anchored) {
            prior *= 0.05; // Heavily penalize jumping in open air
        }
    }
    
    // Food approach logic
    if (state.power_source_count > 0) {
        int hx = snake.parts[0].x, hy = snake.parts[0].y;
        int best_cur = 99999, best_new = 99999;
        for (int f = 0; f < state.power_source_count; f++) {
            int fx = state.power_sources[f].x, fy = state.power_sources[f].y;
            int cd = abs(hx - fx) + abs(hy - fy);
            int nd = abs(nx - fx) + abs(ny - fy);
            if (cd < best_cur) best_cur = cd;
            if (nd < best_new) best_new = nd;
        }
        if (best_new < best_cur) prior += 2.0; 
    }

    return prior;
}

// ─── Corrected UCB Selection ────────────────────────────
Direction duct_select(int node_idx, int snake_idx, const MoveList& moves,
                      const Snakebot& snake, const GameState& state) {
    MCTSNode& node = g_tree[node_idx];
    double priors[5] = {};
    double psum = 0;

    for (int m = 0; m < moves.count; m++) {
        Direction d = moves.moves[m].dir;
        priors[d] = compute_move_prior(snake, d, state);
        psum += priors[d];
    }
    if (psum > 0) for (int m = 0; m < moves.count; m++) priors[moves.moves[m].dir] /= psum;

    double sqrt_total = sqrt((double)(node.total_visits > 0 ? node.total_visits : 1));
    Direction best_d = moves.moves[0].dir;
    double best_ucb = -1e18;

    double parent_q = 0.0;
    int visited_count = 0;
    for (int m = 0; m < moves.count; m++) {
        Direction d = moves.moves[m].dir;
        if (node.visits[snake_idx][d] > 0) {
            parent_q += node.scores[snake_idx][d] / node.visits[snake_idx][d];
            visited_count++;
        }
    }
    double fpu_value = (visited_count > 0) ? (parent_q / visited_count) : 0.0;

    for (int m = 0; m < moves.count; m++) {
        Direction d = moves.moves[m].dir;
        double ucb;
        if (node.visits[snake_idx][d] == 0) {
            ucb = fpu_value + PUCT_C * priors[d] * sqrt_total; 
        } else {
            double q = node.scores[snake_idx][d] / node.visits[snake_idx][d];
            ucb = q + PUCT_C * priors[d] * sqrt_total / (1.0 + node.visits[snake_idx][d]);
        }
        if (ucb > best_ucb) { best_ucb = ucb; best_d = d; }
    }
    return best_d;
}

// ─── True Tree MCTS Main Loop ───────────────────────────
void run_mcts(const GameState& initial_state, Move final_moves[MAX_SNAKES]) {
    auto start_time = hrclock::now();
    int iterations = 0;
    int time_limit_us = (initial_state.turn == 1) ? 950000 : 40000;

    g_node_count = 0;
    int root_node = alloc_node();

    while (true) {
        GameState sim_state;
        fast_copy_state(sim_state, initial_state);

        int path_nodes[DUCT_MAX_DEPTH];
        Direction path_dirs[DUCT_MAX_DEPTH][MAX_SNAKES];
        int actual_depth = 0;
        int curr_node = root_node;

        while (actual_depth < DUCT_MAX_DEPTH) {
            path_nodes[actual_depth] = curr_node;
            build_occupancy(sim_state);

            Move chosen_moves[MAX_SNAKES];
            int joint_key = 0;

            for (int i = 0; i < sim_state.snake_count; i++) {
                if (!sim_state.snakes[i].is_alive) {
                    chosen_moves[i].dir = NONE;
                    path_dirs[actual_depth][i] = NONE;
                    continue;
                }
                MoveList moves = generate_moves(sim_state.snakes[i]);
                Direction d = duct_select(curr_node, i, moves, sim_state.snakes[i], sim_state);
                chosen_moves[i].dir = d;
                path_dirs[actual_depth][i] = d;
                joint_key |= (d << (i * 3)); 
            }

            actual_depth++;
            simulate(sim_state, chosen_moves);

            bool my_alive = false, opp_alive = false;
            for (int i = 0; i < sim_state.snake_count; i++) {
                if (!sim_state.snakes[i].is_alive) continue;
                if (sim_state.snakes[i].is_mine) my_alive = true;
                else opp_alive = true;
            }
            if (!my_alive || !opp_alive || sim_state.power_source_count == 0) break;

            int next_node = -1;
            MCTSNode& cnode = g_tree[curr_node];
            for (int c = 0; c < cnode.child_count; c++) {
                if (cnode.child_keys[c] == joint_key) {
                    next_node = cnode.child_nodes[c]; break;
                }
            }

            if (next_node != -1) {
                curr_node = next_node; 
            } else {
                int new_node = alloc_node();
                if (new_node != -1 && cnode.child_count < CHILD_MAX) {
                    cnode.child_keys[cnode.child_count] = joint_key;
                    cnode.child_nodes[cnode.child_count] = new_node;
                    cnode.child_count++;
                }
                break; 
            }
        }

        double raw_score = evaluate(sim_state);

        for (int d = actual_depth - 1; d >= 0; d--) {
            int n_idx = path_nodes[d];
            MCTSNode& st = g_tree[n_idx];
            st.total_visits++;
            
            double discount = pow(0.95, actual_depth - d); 
            double current_score = raw_score * discount;

            for (int i = 0; i < initial_state.snake_count; i++) {
                Direction dir = path_dirs[d][i];
                if (dir == NONE) continue;
                st.visits[i][dir]++;
                double agent_score = initial_state.snakes[i].is_mine ? current_score : -current_score;
                st.scores[i][dir] += agent_score;
            }
        }
        
        iterations++;
        if ((iterations & 127) == 0) {
            if (get_time_us(start_time) >= time_limit_us) break;
            if (g_node_count >= MAX_NODES - 100) break; 
        }
    }

    for (int i = 0; i < initial_state.snake_count; i++) {
        if (!initial_state.snakes[i].is_alive || !initial_state.snakes[i].is_mine) continue;
        MoveList moves = generate_moves(initial_state.snakes[i]);
        int best_dir = moves.moves[0].dir, max_visits = -1;
        for (int m = 0; m < moves.count; m++) {
            Direction d = moves.moves[m].dir;
            if (g_tree[0].visits[i][d] > max_visits) {
                max_visits = g_tree[0].visits[i][d];
                best_dir = d;
            }
        }
        final_moves[i].dir = static_cast<Direction>(best_dir);
    }
}

// ─── Main ───────────────────────────────────────────────
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    GameState state;
    state.turn = 0;
    state.snake_count = 0;

    int my_id;
    cin >> my_id; cin.ignore();
    cin >> g_width; cin.ignore();
    cin >> g_height; cin.ignore();
    g_total = g_width * g_height;

    memset(g_wall, 0, sizeof(g_wall));
    for (int y = 0; y < g_height; y++) {
        string row; getline(cin, row);
        if ((int)row.size() < g_width) row.resize(g_width, ' ');
        for (int x = 0; x < g_width; x++)
            g_wall[y * g_width + x] = (row[x] == '#');
    }

    int snakebots_per_player;
    cin >> snakebots_per_player; cin.ignore();

    int my_bot_ids[MAX_SNAKES];
    int my_bot_count = 0;
    for (int i = 0; i < snakebots_per_player; i++) {
        int id; cin>>id; cin.ignore();
        my_bot_ids[my_bot_count++] = id;
    }
    for (int i = 0; i < snakebots_per_player; i++) {
        int id; cin>>id; cin.ignore();
    }

    while (1) {
        state.turn++;
        cin >> state.power_source_count;
        if (cin.fail() || cin.eof()) break;
        cin.ignore();
        if (state.power_source_count > MAX_POWER_SOURCES)
            state.power_source_count = MAX_POWER_SOURCES;
        for (int i = 0; i < state.power_source_count; i++) {
            int x, y; cin>>x>>y; cin.ignore();
            state.power_sources[i] = {x, y};
        }

        int incoming_snake_count;
        cin >> incoming_snake_count; cin.ignore();

        for (int i = 0; i < state.snake_count; i++)
            state.snakes[i].is_alive = false;

        for (int i = 0; i < incoming_snake_count; i++) {
            int snakebot_id; string body;
            cin >> snakebot_id >> body; cin.ignore();
            int slot = -1;
            for (int j = 0; j < state.snake_count; j++)
                if (state.snakes[j].id == snakebot_id) { slot = j; break; }
            if (slot == -1 && state.snake_count < MAX_SNAKES)
                slot = state.snake_count++;
            if (slot == -1) continue;
            state.snakes[slot].id       = snakebot_id;
            state.snakes[slot].is_alive = true;
            state.snakes[slot].is_mine  = false;
            for (int k = 0; k < my_bot_count; k++)
                if (snakebot_id == my_bot_ids[k]) { state.snakes[slot].is_mine = true; break; }
            parse_body(body, state.snakes[slot]);
            if (state.snakes[slot].part_count > MAX_PARTS)
                state.snakes[slot].part_count = MAX_PARTS;
        }

        Move final_moves[MAX_SNAKES];
        for (int i = 0; i < MAX_SNAKES; i++) final_moves[i].dir = NONE;
        run_mcts(state, final_moves);

        string output = "";
        bool first = true;
        for (int i = 0; i < state.snake_count; i++) {
            if (state.snakes[i].is_alive && state.snakes[i].is_mine) {
                Direction chosen = final_moves[i].dir;
                if (chosen == NONE) {
                    build_occupancy(state);
                    MoveList ml = generate_moves(state.snakes[i]);
                    chosen = ml.count > 0 ? ml.moves[0].dir : UP;
                }
                if (!first) output += ";";
                output += to_string(state.snakes[i].id) + " " + DIR_NAMES[chosen];
                first = false;
            }
        }
        if (output == "") output = "WAIT";
        cout << output << endl;
    }
    return 0;
}