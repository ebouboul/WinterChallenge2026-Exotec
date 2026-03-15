#undef _GLIBCXX_DEBUG
#pragma GCC optimize("Ofast,inline,unroll-loops,omit-frame-pointer")

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <set>
#include <map>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>

using namespace std;
using namespace std::chrono;

const int MAX_DIM = 50;
const int MAX_BOTS = 16;
const int MAX_APPLES = 200;
const int MAX_BODY = 700;

// ==================== TUNEABLE EVALUATION CONSTANTS ====================
// The Genetic Algorithm injects these at compile time.
#ifndef MAX_SEARCH_DEPTH
#define MAX_SEARCH_DEPTH 10
#endif
#ifndef W_SCORE
#define W_SCORE 0.40000
#endif
#ifndef W_BOT
#define W_BOT 0.30000
#endif
#ifndef W_DIST_MY
#define W_DIST_MY 0.05000
#endif
#ifndef W_DIST_OPP
#define W_DIST_OPP 0.05000
#endif
#ifndef W_SAFETY
#define W_SAFETY 0.10000
#endif
#ifndef W_GROUND
#define W_GROUND 0.15000
#endif
#ifndef W_MOB
#define W_MOB 0.02000
#endif
#ifndef W_TERR
#define W_TERR 0.15000
#endif
#ifndef UCB_EXPLORATION
#define UCB_EXPLORATION 0.50000
#endif
// =======================================================================

const int DX[] = {0, 0, -1, 1};
const int DY[] = {-1, 1, 0, 0};
const int OPPOSITE[] = {1, 0, 3, 2};
const string DIR_NAMES[] = {"UP", "DOWN", "LEFT", "RIGHT"};

int W, H;
bool globalIsPlatform[MAX_DIM][MAX_DIM];

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
};

int getFacing(const Point* body, int body_size) {
    if (body_size < 2) return -1;
    int dx = body[0].x - body[1].x;
    int dy = body[0].y - body[1].y;
    if (dx == 0 && dy == -1) return 0;
    if (dx == 0 && dy == 1)  return 1;
    if (dx == -1 && dy == 0) return 2;
    if (dx == 1 && dy == 0)  return 3;
    return -1;
}

int getLegalMoves(const Point* body, int body_size, int legal[4]) {
    int facing = getFacing(body, body_size);
    int count = 0;
    for (int d = 0; d < 4; d++) {
        if (facing >= 0 && d == OPPOSITE[facing]) continue;
        legal[count++] = d;
    }
    if (count == 0) { for (int d = 0; d < 4; d++) legal[count++] = d; }
    return count;
}

struct Snakebot {
    int id;
    Point _body[MAX_BODY + MAX_SEARCH_DEPTH];
    int head_idx;
    int body_size;
    int dir;
    int owner;
    bool alive;

    Point* body() { return &_body[head_idx]; }
    const Point* body() const { return &_body[head_idx]; }

    void push_front(Point p) {
        head_idx--;
        _body[head_idx] = p;
        if (body_size < MAX_BODY) body_size++;
    }
    void pop_back() { if (body_size > 0) body_size--; }
    void erase_head() {
        if (body_size > 0) {
            head_idx++;
            body_size--;
        }
    }
    int currentFacing() const { return getFacing(body(), body_size); }
};

struct GameState {
    Point apples[MAX_APPLES];
    int num_apples;
    Snakebot bots[MAX_BOTS];
    int num_bots;
    int turn;
    int losses[2] = {0, 0};

    void doMoves(const int actions[], int actionCount) {
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            int d = (i < actionCount) ? actions[i] : -1;
            if (d < 0 || d >= 4) {
                d = bots[i].currentFacing();
                if (d < 0) d = 0;
            }
            int facing = bots[i].currentFacing();
            if (facing >= 0 && d == OPPOSITE[facing]) d = facing;
            bots[i].dir = d;

            Point newHead = {bots[i].body()[0].x + DX[d], bots[i].body()[0].y + DY[d]};
            bool willEatApple = false;
            for (int a = 0; a < num_apples; a++) {
                if (apples[a] == newHead) { willEatApple = true; break; }
            }
            if (!willEatApple) bots[i].pop_back();
            bots[i].push_front(newHead);
        }
    }

    void doEats() {
        bool eaten[MAX_APPLES] = {};
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            Point head = bots[i].body()[0];
            for (int a = 0; a < num_apples; a++) {
                if (apples[a] == head) eaten[a] = true;
            }
        }
        int n = 0;
        for (int a = 0; a < num_apples; a++)
            if (!eaten[a]) apples[n++] = apples[a];
        num_apples = n;
    }

    void doBeheadings() {
        int8_t bodyOwner[MAX_DIM][MAX_DIM]; 
        memset(bodyOwner, -1, sizeof(bodyOwner));
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            for (int k = 1; k < bots[i].body_size; k++) {
                int bx = bots[i].body()[k].x, by = bots[i].body()[k].y;
                if (bx >= 0 && bx < W && by >= 0 && by < H)
                    bodyOwner[by][bx] = (int8_t)i;
            }
        }

        int toBehead[MAX_BOTS]; int numToBehead = 0;
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            Point head = bots[i].body()[0];
            if (head.x < 0 || head.x >= W || head.y < 0 || head.y >= H) {
                toBehead[numToBehead++] = i; continue;
            }
            if (globalIsPlatform[head.y][head.x]) {
                toBehead[numToBehead++] = i; continue;
            }
            if (bodyOwner[head.y][head.x] >= 0) {
                toBehead[numToBehead++] = i; continue;
            }
            for (int j = 0; j < num_bots; j++) {
                if (j == i || !bots[j].alive) continue;
                if (bots[j].body()[0] == head) { toBehead[numToBehead++] = i; break; }
            }
        }
        for (int idx = 0; idx < numToBehead; idx++) {
            int i = toBehead[idx];
            if (bots[i].body_size <= 3) {
                losses[bots[i].owner] += bots[i].body_size;
                bots[i].alive = false;
            } else {
                losses[bots[i].owner]++;
                bots[i].erase_head();
            }
        }
    }

    bool somethingSolidUnder(Point c, const Point* ignoreBody, int ignoreSize,
                             const bool solidGrid[MAX_DIM][MAX_DIM]) const {
        Point below = {c.x, c.y + 1};
        if (below.x < 0 || below.x >= W || below.y < 0 || below.y >= H) return false;
        if (!solidGrid[below.y][below.x]) return false;
        for (int i = 0; i < ignoreSize; i++)
            if (ignoreBody[i] == below) return false;
        return true;
    }

    void buildSolidGrid(bool solidGrid[MAX_DIM][MAX_DIM]) const {
        memcpy(solidGrid, globalIsPlatform, sizeof(globalIsPlatform));
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            for (int j = 0; j < bots[i].body_size; j++) {
                int bx = bots[i].body()[j].x, by = bots[i].body()[j].y;
                if (bx >= 0 && bx < W && by >= 0 && by < H) solidGrid[by][bx] = true;
            }
        }
        for (int a = 0; a < num_apples; a++) {
            int ax = apples[a].x, ay = apples[a].y;
            if (ax >= 0 && ax < W && ay >= 0 && ay < H) solidGrid[ay][ax] = true;
        }
    }

    void doFalls() {
        bool somethingFell = true;
        while (somethingFell) {
            bool solidGrid[MAX_DIM][MAX_DIM];
            buildSolidGrid(solidGrid);
            while (somethingFell) {
                somethingFell = false;
                for (int i = 0; i < num_bots; i++) {
                    if (!bots[i].alive) continue;
                    bool canFall = true;
                    for (int k = 0; k < bots[i].body_size; k++) {
                        if (somethingSolidUnder(bots[i].body()[k], bots[i].body(), bots[i].body_size, solidGrid)) {
                            canFall = false; break;
                        }
                    }
                    if (canFall) {
                        somethingFell = true;
                        for (int k = 0; k < bots[i].body_size; k++) {
                            int ox = bots[i].body()[k].x, oy = bots[i].body()[k].y;
                            if (ox >= 0 && ox < W && oy >= 0 && oy < H) solidGrid[oy][ox] = false;
                        }
                        for (int k = 0; k < bots[i].body_size; k++) bots[i].body()[k].y++;
                        for (int k = 0; k < bots[i].body_size; k++) {
                            int nx = bots[i].body()[k].x, ny = bots[i].body()[k].y;
                            if (nx >= 0 && nx < W && ny >= 0 && ny < H) solidGrid[ny][nx] = true;
                        }
                        bool allOOB = true;
                        for (int k = 0; k < bots[i].body_size; k++) {
                            if (bots[i].body()[k].y < H + 1) { allOOB = false; break; }
                        }
                        if (allOOB) bots[i].alive = false;
                    }
                }
            }
            somethingFell = doIntercoiledFalls();
        }
    }

    bool birdsAreTouching(int i, int j) const {
        for (int k1 = 0; k1 < bots[i].body_size; k1++)
            for (int k2 = 0; k2 < bots[j].body_size; k2++)
                if (abs(bots[i].body()[k1].x - bots[j].body()[k2].x) +
                    abs(bots[i].body()[k1].y - bots[j].body()[k2].y) == 1)
                    return true;
        return false;
    }

    bool doIntercoiledFalls() {
        bool fell = false, somethingFell = true;
        while (somethingFell) {
            somethingFell = false;
            bool solidGrid[MAX_DIM][MAX_DIM];
            buildSolidGrid(solidGrid);
            bool visited[MAX_BOTS] = {};
            for (int i = 0; i < num_bots; i++) {
                if (!bots[i].alive || visited[i]) continue;
                int group[MAX_BOTS], gsz = 0;
                int q[MAX_BOTS]; int qh = 0, qt = 0;
                q[qt++] = i; visited[i] = true;
                while (qh < qt) {
                    int cur = q[qh++]; group[gsz++] = cur;
                    for (int j = 0; j < num_bots; j++) {
                        if (j == cur || !bots[j].alive || visited[j]) continue;
                        if (birdsAreTouching(cur, j)) { visited[j] = true; q[qt++] = j; }
                    }
                }
                if (gsz <= 1) continue;
                Point metaBody[MAX_BODY * 2]; int metaSize = 0;
                for (int gi = 0; gi < gsz; gi++)
                    for (int k = 0; k < bots[group[gi]].body_size; k++)
                        if (metaSize < MAX_BODY * 2) metaBody[metaSize++] = bots[group[gi]].body()[k];
                bool canFall = true;
                for (int mi = 0; mi < metaSize && canFall; mi++)
                    if (somethingSolidUnder(metaBody[mi], metaBody, metaSize, solidGrid)) canFall = false;
                if (canFall) {
                    somethingFell = true; fell = true;
                    for (int gi = 0; gi < gsz; gi++) {
                        int bi = group[gi];
                        for (int k = 0; k < bots[bi].body_size; k++) bots[bi].body()[k].y++;
                        if (bots[bi].body()[0].y >= H) bots[bi].alive = false;
                    }
                }
            }
        }
        return fell;
    }

    void simulate(const int actions[], int actionCount) {
        doMoves(actions, actionCount);
        doEats();
        doBeheadings();
        doFalls();
        turn++;
    }

    int score(int pIdx) const {
        int s = 0;
        for (int i = 0; i < num_bots; i++)
            if (bots[i].alive && bots[i].owner == pIdx) s += bots[i].body_size;
        return s;
    }

    bool isTerminal() const {
        if (turn >= 200 || num_apples == 0) return true;
        bool p0 = false, p1 = false;
        for (int i = 0; i < num_bots; i++) {
            if (!bots[i].alive) continue;
            if (bots[i].owner == 0) p0 = true;
            if (bots[i].owner == 1) p1 = true;
        }
        return !p0 || !p1;
    }

    int getPlayerBots(int pIdx, int indices[MAX_BOTS]) const {
        int n = 0;
        for (int i = 0; i < num_bots; i++)
            if (bots[i].alive && bots[i].owner == pIdx) indices[n++] = i;
        return n;
    }
};

// ==================== STATIC EVALUATION (Heuristic) ====================

static int floodFillCount(Point start, const GameState& s) {
    const int CAP = 20;
    bool visited[MAX_DIM][MAX_DIM];
    memset(visited, 0, sizeof(visited));
    bool blocked[MAX_DIM][MAX_DIM];
    memset(blocked, 0, sizeof(blocked));
    for (int i = 0; i < s.num_bots; i++) {
        if (!s.bots[i].alive) continue;
        for (int j = 0; j < s.bots[i].body_size; j++) {
            int bx = s.bots[i].body()[j].x, by = s.bots[i].body()[j].y;
            if (bx >= 0 && bx < W && by >= 0 && by < H) blocked[by][bx] = true;
        }
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (globalIsPlatform[y][x]) blocked[y][x] = true;
            
    Point q[MAX_DIM * MAX_DIM];
    int qh = 0, qt = 0, count = 0;
    if (start.x >= 0 && start.x < W && start.y >= 0 && start.y < H && !blocked[start.y][start.x]) {
        q[qt++] = start; visited[start.y][start.x] = true;
    }
    while (qh < qt && count < CAP) {
        Point cur = q[qh++]; count++;
        for (int d = 0; d < 4; d++) {
            int nx = cur.x + DX[d], ny = cur.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !visited[ny][nx] && !blocked[ny][nx]) {
                visited[ny][nx] = true; q[qt++] = {nx, ny};
            }
        }
    }
    return count;
}

double evaluate(const GameState& s, int myP) {
    int myIdx[MAX_BOTS], oppIdx[MAX_BOTS];
    int nMy = s.getPlayerBots(myP, myIdx);
    int nOpp = s.getPlayerBots(1 - myP, oppIdx);

    double scoreDiff = (double)(s.score(myP) - s.score(1 - myP));
    double botDiff   = (double)(nMy - nOpp);
    
    double myDist = 0, oppDist = 0;
    for (int i = 0; i < nMy; i++) {
        int best = 9999; Point h = s.bots[myIdx[i]].body()[0];
        for (int a = 0; a < s.num_apples; a++) {
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < best) best = d;
        }
        myDist += (best < 9999 ? best : 0);
    }
    for (int i = 0; i < nOpp; i++) {
        int best = 9999; Point h = s.bots[oppIdx[i]].body()[0];
        for (int a = 0; a < s.num_apples; a++) {
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < best) best = d;
        }
        oppDist += (best < 9999 ? best : 0);
    }
    double avgMyDist  = nMy  > 0 ? myDist  / nMy  : 0.0;
    double avgOppDist = nOpp > 0 ? oppDist / nOpp : 0.0;

    bool bodyGrid[MAX_DIM][MAX_DIM];
    memset(bodyGrid, 0, sizeof(bodyGrid));
    for (int i = 0; i < s.num_bots; i++) {
        if (!s.bots[i].alive) continue;
        for (int j = 0; j < s.bots[i].body_size; j++) {
            int bx = s.bots[i].body()[j].x, by = s.bots[i].body()[j].y;
            if (bx >= 0 && bx < W && by >= 0 && by < H) bodyGrid[by][bx] = true;
        }
    }
    
    double mySafety = 0, oppSafety = 0;
    for (int i = 0; i < nMy; i++) {
        Point h = s.bots[myIdx[i]].body()[0]; int safe = 0;
        for (int d = 0; d < 4; d++) {
            int nx = h.x + DX[d], ny = h.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !globalIsPlatform[ny][nx] && !bodyGrid[ny][nx]) safe++;
        }
        mySafety += safe;
    }
    for (int i = 0; i < nOpp; i++) {
        Point h = s.bots[oppIdx[i]].body()[0]; int safe = 0;
        for (int d = 0; d < 4; d++) {
            int nx = h.x + DX[d], ny = h.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !globalIsPlatform[ny][nx] && !bodyGrid[ny][nx]) safe++;
        }
        oppSafety += safe;
    }
    double avgMySafety  = nMy  > 0 ? mySafety  / nMy  : 0.0;
    double avgOppSafety = nOpp > 0 ? oppSafety / nOpp : 0.0;

    double myGround = 0, oppGround = 0;
    for (int i = 0; i < nMy; i++) {
        Point h = s.bots[myIdx[i]].body()[0]; int by = h.y + 1;
        if (by < H && (globalIsPlatform[by][h.x] || bodyGrid[by][h.x])) myGround += 1.0;
        else if (by >= H) myGround += 0.5;
    }
    for (int i = 0; i < nOpp; i++) {
        Point h = s.bots[oppIdx[i]].body()[0]; int by = h.y + 1;
        if (by < H && (globalIsPlatform[by][h.x] || bodyGrid[by][h.x])) oppGround += 1.0;
        else if (by >= H) oppGround += 0.5;
    }
    double avgMyGround  = nMy  > 0 ? myGround  / nMy  : 0.0;
    double avgOppGround = nOpp > 0 ? oppGround / nOpp : 0.0;

    double myMobility = 0, oppMobility = 0;
    for (int i = 0; i < nMy; i++) myMobility += floodFillCount(s.bots[myIdx[i]].body()[0], s);
    for (int i = 0; i < nOpp; i++) oppMobility += floodFillCount(s.bots[oppIdx[i]].body()[0], s);
    double avgMyMob  = nMy  > 0 ? myMobility  / nMy  : 0.0;
    double avgOppMob = nOpp > 0 ? oppMobility / nOpp : 0.0;

    double myOwned = 0, oppOwned = 0;
    for (int a = 0; a < s.num_apples; a++) {
        int bmd = 9999, bod = 9999;
        for (int i = 0; i < nMy; i++) {
            Point h = s.bots[myIdx[i]].body()[0];
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < bmd) bmd = d;
        }
        for (int i = 0; i < nOpp; i++) {
            Point h = s.bots[oppIdx[i]].body()[0];
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < bod) bod = d;
        }
        if (bmd < bod) myOwned += 1.0;
        else if (bod < bmd) oppOwned += 1.0;
    }

    // TUNEABLE MACRO MATH (No Neural Network)
    double logit = W_SCORE * scoreDiff
                 + W_BOT * botDiff
                 - W_DIST_MY * avgMyDist
                 + W_DIST_OPP * avgOppDist
                 + W_SAFETY * (avgMySafety - avgOppSafety)
                 + W_GROUND * (avgMyGround - avgOppGround)
                 + W_MOB * (avgMyMob - avgOppMob)
                 + W_TERR * (myOwned - oppOwned);

    return 1.0 / (1.0 + exp(-logit));
}

// ==================== LOOKUP GRIDS ====================
bool gHasBody[MAX_DIM][MAX_DIM];
bool gHasApple[MAX_DIM][MAX_DIM];

void buildLookups(const GameState& state) {
    memset(gHasBody, 0, sizeof(gHasBody));
    memset(gHasApple, 0, sizeof(gHasApple));
    for (int i = 0; i < state.num_bots; i++) {
        if (!state.bots[i].alive) continue;
        for (int j = 0; j < state.bots[i].body_size; j++) {
            int bx = state.bots[i].body()[j].x, by = state.bots[i].body()[j].y;
            if (bx >= 0 && bx < W && by >= 0 && by < H) gHasBody[by][bx] = true;
        }
    }
    for (int a = 0; a < state.num_apples; a++) {
        int ax = state.apples[a].x, ay = state.apples[a].y;
        if (ax >= 0 && ax < W && ay >= 0 && ay < H) gHasApple[ay][ax] = true;
    }
}

int greedyBotAction(const Snakebot& bot, const GameState& state) {
    Point head = bot.body()[0];
    int legal[4], nLegal = getLegalMoves(bot.body(), bot.body_size, legal);
    int bestDist = 9999; Point bestTarget = head;
    for (int a = 0; a < state.num_apples; a++) {
        int d = abs(state.apples[a].x - head.x) + abs(state.apples[a].y - head.y);
        if (d < bestDist) { bestDist = d; bestTarget = state.apples[a]; }
    }
    if (state.num_apples == 0) return legal[rng() % nLegal];
    int bestDir = legal[0], bestScore = -10000;
    for (int li = 0; li < nLegal; li++) {
        int d = legal[li];
        int nx = head.x + DX[d], ny = head.y + DY[d];
        int score = 0;
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) score -= 500;
        else {
            if (globalIsPlatform[ny][nx] || (gHasBody[ny][nx] && !gHasApple[ny][nx])) score -= 500;
            if (gHasApple[ny][nx]) score += 200;
            score -= (abs(bestTarget.x - nx) + abs(bestTarget.y - ny)) * 2;
            if (ny + 1 < H && globalIsPlatform[ny + 1][nx]) score += 10;
        }
        if (score > bestScore) { bestScore = score; bestDir = d; }
    }
    return bestDir;
}

struct BotStats {
    double totalReward[4] = {}; 
    int visits[4] = {};
};

int selectUCB(const BotStats& stats, int parentVisits, const double biases[4],
              const int legal[], int nLegal) {
    int best = legal[0];
    double bestVal = -1e18;
    for (int li = 0; li < nLegal; li++) {
        int d = legal[li];
        int n = stats.visits[d];
        double val;
        if (n == 0) {
            val = 1e9 + biases[d];
        } else {
            double exploitation = stats.totalReward[d] / n;
            double exploration = UCB_EXPLORATION * sqrt(log((double)max(parentVisits, 1)) / n);
            double progressiveBias = biases[d] / (n + 1);
            val = exploitation + exploration + progressiveBias;
        }
        if (val > bestVal) { bestVal = val; best = d; }
    }
    return best;
}

int selectMostVisited(const BotStats& stats, const int legal[], int nLegal) {
    int best = legal[0], bestN = -1;
    for (int li = 0; li < nLegal; li++) {
        int d = legal[li];
        if (stats.visits[d] > bestN) { bestN = stats.visits[d]; best = d; }
    }
    return best;
}

void computeBiases(const Snakebot& bot, const GameState& state, double biases[4]) {
    Point head = bot.body()[0];
    int bestDist = 9999;
    for (int a = 0; a < state.num_apples; a++) {
        int d = abs(state.apples[a].x - head.x) + abs(state.apples[a].y - head.y);
        if (d < bestDist) bestDist = d;
    }
    for (int d = 0; d < 4; d++) {
        biases[d] = 0.0;
        int nx = head.x + DX[d], ny = head.y + DY[d];
        if (nx < 0 || nx >= W || ny < 0 || ny >= H) { biases[d] = -0.5; continue; }
        if (globalIsPlatform[ny][nx]) { biases[d] = -0.5; continue; }
        if (gHasBody[ny][nx] && !gHasApple[ny][nx]) { biases[d] = -0.3; continue; }
        if (gHasApple[ny][nx]) { biases[d] = 0.5; continue; }
        if (state.num_apples > 0) {
            int newBest = 9999;
            for (int a = 0; a < state.num_apples; a++) {
                int dd = abs(state.apples[a].x - nx) + abs(state.apples[a].y - ny);
                if (dd < newBest) newBest = dd;
            }
            if (newBest < bestDist) biases[d] += 0.2;
            else if (newBest > bestDist) biases[d] -= 0.05;
        }
        if (ny + 1 < H && (globalIsPlatform[ny + 1][nx] || gHasApple[ny + 1][nx] || gHasBody[ny + 1][nx]))
            biases[d] += 0.1;
        else if (ny + 1 >= H)
            biases[d] -= 0.2;
        if (d == 0) biases[d] -= 0.05;
    }
}

struct DepthNode {
    BotStats targetStats;
    int totalVisits = 0;
};

struct SingleBotResult {
    int botId;
    int botDir;
};

SingleBotResult runMCTSForSingleBot(const GameState& rootState, int myPlayer,
                                     int targetBotGlobalIdx, const int lockedMoves[],
                                     int timeLimitMs) {
    auto startTime = high_resolution_clock::now();
    buildLookups(rootState);

    int myIdx[MAX_BOTS], oppIdx[MAX_BOTS];
    int nMy = rootState.getPlayerBots(myPlayer, myIdx);
    int nOpp = rootState.getPlayerBots(1 - myPlayer, oppIdx);

    SingleBotResult res;
    res.botId = rootState.bots[targetBotGlobalIdx].id;
    res.botDir = 0;

    int targetLocalIdx = -1;
    for (int i = 0; i < nMy; i++) {
        if (myIdx[i] == targetBotGlobalIdx) { targetLocalIdx = i; break; }
    }
    if (targetLocalIdx < 0) return res;

    unordered_map<uint64_t, DepthNode> tree[MAX_SEARCH_DEPTH];
    for (int d = 0; d < MAX_SEARCH_DEPTH; d++) tree[d].reserve(128);

    double rootTargetBiases[4];
    int rootTargetLegal[4], rootTargetNLegal;
    computeBiases(rootState.bots[targetBotGlobalIdx], rootState, rootTargetBiases);
    rootTargetNLegal = getLegalMoves(rootState.bots[targetBotGlobalIdx].body(),
                                     rootState.bots[targetBotGlobalIdx].body_size, rootTargetLegal);

    int totalIter = 0;

    while (true) {
        if ((totalIter & 15) == 0) {
            if (duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count() >= timeLimitMs)
                break;
        }
        totalIter++;

        int pathTargetAct[MAX_SEARCH_DEPTH];
        uint64_t pathHashes[MAX_SEARCH_DEPTH];
        int pathDepth = 0;

        GameState currentState = rootState;
        uint64_t runningHash = 0;

        for (int depth = 0; depth < MAX_SEARCH_DEPTH; depth++) {
            if (currentState.isTerminal()) break;

            auto& node = tree[depth][runningHash];
            node.totalVisits++;

            int curMyIdx[MAX_BOTS], curOppIdx[MAX_BOTS];
            int curNMy = currentState.getPlayerBots(myPlayer, curMyIdx);
            int curNOpp = currentState.getPlayerBots(1 - myPlayer, curOppIdx);
            if (curNMy == 0) break;

            double targetBiases[4];
            int targetLegal[4], targetNLeg;

            int curTargetLocalIdx = -1;
            for (int i = 0; i < curNMy; i++) {
                if (curMyIdx[i] == targetBotGlobalIdx) { curTargetLocalIdx = i; break; }
            }

            if (depth == 0) {
                memcpy(targetBiases, rootTargetBiases, sizeof(rootTargetBiases));
                memcpy(targetLegal, rootTargetLegal, sizeof(rootTargetLegal));
                targetNLeg = rootTargetNLegal;
            } else {
                buildLookups(currentState);
                if (curTargetLocalIdx >= 0) {
                    computeBiases(currentState.bots[targetBotGlobalIdx], currentState, targetBiases);
                    targetNLeg = getLegalMoves(currentState.bots[targetBotGlobalIdx].body(),
                                               currentState.bots[targetBotGlobalIdx].body_size, targetLegal);
                }
            }

            int allActions[MAX_BOTS];
            memset(allActions, -1, sizeof(allActions));
            int targetAct = -1;

            for (int i = 0; i < curNMy; i++) {
                int botGlobalIdx = curMyIdx[i];
                if (botGlobalIdx == targetBotGlobalIdx) {
                    targetAct = selectUCB(node.targetStats, node.totalVisits,
                                           targetBiases, targetLegal, targetNLeg);
                    allActions[botGlobalIdx] = targetAct;
                } else if (lockedMoves[botGlobalIdx] != -1) {
                    if (depth == 0) {
                        allActions[botGlobalIdx] = lockedMoves[botGlobalIdx];
                    } else {
                        allActions[botGlobalIdx] = greedyBotAction(currentState.bots[botGlobalIdx], currentState);
                    }
                } else {
                    allActions[botGlobalIdx] = greedyBotAction(currentState.bots[botGlobalIdx], currentState);
                }
            }

            for (int i = 0; i < curNOpp; i++) {
                int oppGI = curOppIdx[i];
                if ((rng() % 4) == 0) {
                    int oppLegal[4];
                    int nOppLegal = getLegalMoves(currentState.bots[oppGI].body(),
                                                  currentState.bots[oppGI].body_size, oppLegal);
                    allActions[oppGI] = oppLegal[rng() % nOppLegal];
                } else {
                    allActions[oppGI] = greedyBotAction(currentState.bots[oppGI], currentState);
                }
            }

            if (targetAct < 0) break;

            pathTargetAct[depth] = targetAct;
            pathHashes[depth] = runningHash;
            pathDepth = depth + 1;

            runningHash = runningHash * 131 + (uint64_t)(targetAct + 1);
            currentState.simulate(allActions, currentState.num_bots);
        }

        double result;
        if (currentState.isTerminal()) {
            int ms = currentState.score(myPlayer), os = currentState.score(1 - myPlayer);
            if (ms == os) {
                ms -= currentState.losses[myPlayer];
                os -= currentState.losses[1 - myPlayer];
            }
            if (ms > os) result = 1.0;
            else if (ms < os) result = 0.0;
            else result = 0.5;
        } else {
            result = evaluate(currentState, myPlayer);
        }

        for (int depth = 0; depth < pathDepth; depth++) {
            auto it = tree[depth].find(pathHashes[depth]);
            if (it == tree[depth].end()) continue;
            auto& node = it->second;

            int d = pathTargetAct[depth];
            if (d >= 0 && d < 4) {
                node.targetStats.totalReward[d] += result;
                node.targetStats.visits[d]++;
            }
        }
    }

    auto it = tree[0].find(0);
    if (it != tree[0].end()) {
        res.botDir = selectMostVisited(it->second.targetStats, rootTargetLegal, rootTargetNLegal);
    } else {
        res.botDir = rootTargetLegal[0];
    }

    return res;
}

struct FinalMove { int botId; int dir; };

int main() {
    int myId;
    cin >> myId >> W >> H; cin.ignore();

    memset(globalIsPlatform, 0, sizeof(globalIsPlatform));
    for (int y = 0; y < H; y++) {
        string row; getline(cin, row);
        for (int x = 0; x < W; x++) if (row[x] == '#') globalIsPlatform[y][x] = true;
    }

    int snakebotsPerPlayer;
    cin >> snakebotsPerPlayer; cin.ignore();
    set<int> myIdSet;
    for (int i = 0; i < snakebotsPerPlayer; i++) {
        int id; cin >> id; cin.ignore(); myIdSet.insert(id);
    }
    for (int i = 0; i < snakebotsPerPlayer; i++) {
        int id; cin >> id; cin.ignore();
    }

    int turnNum = 0;
    bool firstTurn = true;

    while (1) {
        GameState state;
        state.turn = turnNum;

        cin >> state.num_apples; cin.ignore();
        for (int i = 0; i < state.num_apples; i++) {
            cin >> state.apples[i].x >> state.apples[i].y; cin.ignore();
        }
        cin >> state.num_bots; cin.ignore();
        for (int i = 0; i < state.num_bots; i++) {
            int sid; string s;
            cin >> sid >> s; cin.ignore();
            state.bots[i].id = sid;
            state.bots[i].owner = myIdSet.count(sid) ? myId : (1 - myId);
            state.bots[i].alive = true;
            state.bots[i].dir = -1;
            state.bots[i].head_idx = MAX_SEARCH_DEPTH;
            state.bots[i].body_size = 0;
            stringstream ss(s); string token;
            while (getline(ss, token, ':') && state.bots[i].body_size < MAX_BODY) {
                int cx, cy;
                sscanf(token.c_str(), "%d,%d", &cx, &cy);
                state.bots[i]._body[state.bots[i].head_idx + state.bots[i].body_size++] = {cx, cy};
            }
        }

        int myBotIdx[MAX_BOTS];
        int nMyBots = state.getPlayerBots(myId, myBotIdx);
        if (nMyBots == 0) {
            cout << "WAIT" << endl;
            turnNum++; firstTurn = false;
            continue;
        }

        sort(myBotIdx, myBotIdx + nMyBots, [&](int a, int b) {
            return state.bots[a].body_size > state.bots[b].body_size; 
        });

        int timeLimit = firstTurn ? 900 : 45;
        int timePerBot = timeLimit / max(nMyBots, 1);

        int lockedMoves[MAX_BOTS];
        memset(lockedMoves, -1, sizeof(lockedMoves));

        vector<FinalMove> finalMoves;

        for (int i = 0; i < nMyBots; i++) {
            int targetGlobalIdx = myBotIdx[i];
            SingleBotResult r = runMCTSForSingleBot(state, myId, targetGlobalIdx,
                                                     lockedMoves, timePerBot);
            lockedMoves[targetGlobalIdx] = r.botDir;
            finalMoves.push_back({r.botId, r.botDir});
        }

        for (size_t i = 0; i < finalMoves.size(); i++) {
            if (i > 0) cout << ";";
            cout << finalMoves[i].botId << " " << DIR_NAMES[finalMoves[i].dir];
        }
        cout << endl;
        turnNum++; firstTurn = false;
    }
}