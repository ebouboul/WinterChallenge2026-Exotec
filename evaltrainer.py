"""
Neural Network Evaluation Trainer for Snakebird Bot
====================================================
Self-play NN training pipeline:
  1. Run generate_data.py first to create dataset.csv from strong bot self-play
  2. Run this script to train the NN and export C++ weights
  3. Paste the output into bot.cpp

Architecture: 8 → 32 → 16 → 1 (ReLU, ReLU, Sigmoid)
Features: score_diff, bot_diff, avg_my_dist, avg_opp_dist,
          safety_diff, ground_diff, mobility_diff, territory_diff
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import csv
import sys

FEATURE_NAMES = [
    "score_diff", "bot_diff", "avg_my_dist", "avg_opp_dist",
    "safety_diff", "ground_diff", "mobility_diff", "territory_diff"
]
NUM_FEATURES = 8

# ==================== 1. LOAD DATA ====================
print("Loading dataset.csv...")
features = []
labels = []
with open("dataset.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) == NUM_FEATURES + 1:
            features.append([float(x) for x in row[:NUM_FEATURES]])
            labels.append(float(row[NUM_FEATURES]))

X = np.array(features, dtype=np.float32)
Y = np.array(labels, dtype=np.float32).reshape(-1, 1)
print(f"Loaded {len(X)} samples with {NUM_FEATURES} features")

# Feature statistics for normalization
means = X.mean(axis=0)
stds = X.std(axis=0)
stds[stds < 1e-6] = 1.0  # avoid division by zero

print("\nFeature statistics:")
for i, name in enumerate(FEATURE_NAMES):
    print(f"  {name:18s} mean={means[i]:8.3f}  std={stds[i]:8.3f}")

# Normalize features (zero mean, unit variance)
X_norm = (X - means) / stds

# Train/test split (80/20)
n = len(X_norm)
idx = np.random.RandomState(42).permutation(n)
split = int(n * 0.8)
X_train = torch.FloatTensor(X_norm[idx[:split]])
Y_train = torch.FloatTensor(Y[idx[:split]])
X_test = torch.FloatTensor(X_norm[idx[split:]])
Y_test = torch.FloatTensor(Y[idx[split:]])

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)

print(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

# ==================== 2. DEFINE NETWORK ====================
class SnakebirdNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SnakebirdNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

# ==================== 3. TRAIN ====================
epochs = 900
best_acc = 0
best_state = None

print("\nTraining...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_binary = (test_pred >= 0.5).float()
            acc = (test_binary == Y_test).float().mean().item()
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}% | Best: {best_acc*100:.2f}%")

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

print(f"\nTraining Complete! Best Test Accuracy: {best_acc*100:.2f}%")

# ==================== 4. EXPORT TO C++ ====================
w1 = model.fc1.weight.detach().numpy()  # [32, 8]
b1 = model.fc1.bias.detach().numpy()    # [32]
w2 = model.fc2.weight.detach().numpy()  # [16, 32]
b2 = model.fc2.bias.detach().numpy()    # [16]
w3 = model.fc3.weight.detach().numpy()  # [1, 16]
b3 = model.fc3.bias.detach().numpy()    # [1]

print("\n" + "="*70)
print("PASTE THE FOLLOWING INTO bot.cpp (replace the evaluate function):")
print("="*70)

cpp = f"""
// ==================== NEURAL NETWORK WEIGHTS (auto-generated) ====================
// Architecture: {NUM_FEATURES} → 32 → 16 → 1 (ReLU, ReLU, Sigmoid)
// Test Accuracy: {best_acc*100:.2f}%
// Features: {', '.join(FEATURE_NAMES)}

// Feature normalization constants
constexpr float NN_MEAN[{NUM_FEATURES}] = {{ {','.join([f"{m:.5f}f" for m in means])} }};
constexpr float NN_STD[{NUM_FEATURES}] = {{ {','.join([f"{s:.5f}f" for s in stds])} }};

// Layer 1: {NUM_FEATURES} → 32
constexpr float NN_W1[32][{NUM_FEATURES}] = {{
{','.join(['    {' + ','.join([f"{val:.5f}f" for val in row]) + '}' for row in w1])}
}};
constexpr float NN_B1[32] = {{ {','.join([f"{val:.5f}f" for val in b1])} }};

// Layer 2: 32 → 16
constexpr float NN_W2[16][32] = {{
{','.join(['    {' + ','.join([f"{val:.5f}f" for val in row]) + '}' for row in w2])}
}};
constexpr float NN_B2[16] = {{ {','.join([f"{val:.5f}f" for val in b2])} }};

// Layer 3: 16 → 1
constexpr float NN_W3[16] = {{ {','.join([f"{val:.5f}f" for val in w3[0]])} }};
constexpr float NN_B3 = {b3[0]:.5f}f;
// =================================================================================
"""
print(cpp)

# Also save to file for easy copy
with open("nn_weights.h", "w") as f:
    f.write(cpp)

# Generate the full evaluate function
eval_func = f"""
// ==================== EVALUATION (Neural Network) ====================

static int floodFillCount(Point start, const GameState& s) {{
    const int CAP = 20;
    bool visited[MAX_DIM][MAX_DIM];
    memset(visited, 0, sizeof(visited));
    bool blocked[MAX_DIM][MAX_DIM];
    memset(blocked, 0, sizeof(blocked));
    for (int i = 0; i < s.num_bots; i++) {{
        if (!s.bots[i].alive) continue;
        for (int j = 0; j < s.bots[i].body_size; j++) {{
            int bx = s.bots[i].body()[j].x, by = s.bots[i].body()[j].y;
            if (bx >= 0 && bx < W && by >= 0 && by < H) blocked[by][bx] = true;
        }}
    }}
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            if (globalIsPlatform[y][x]) blocked[y][x] = true;
    Point q[MAX_DIM * MAX_DIM];
    int qh = 0, qt = 0, count = 0;
    if (start.x >= 0 && start.x < W && start.y >= 0 && start.y < H && !blocked[start.y][start.x]) {{
        q[qt++] = start; visited[start.y][start.x] = true;
    }}
    while (qh < qt && count < CAP) {{
        Point cur = q[qh++]; count++;
        for (int d = 0; d < 4; d++) {{
            int nx = cur.x + DX[d], ny = cur.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !visited[ny][nx] && !blocked[ny][nx]) {{
                visited[ny][nx] = true; q[qt++] = {{nx, ny}};
            }}
        }}
    }}
    return count;
}}

double evaluate(const GameState& s, int myP) {{
    int myIdx[MAX_BOTS], oppIdx[MAX_BOTS];
    int nMy = s.getPlayerBots(myP, myIdx);
    int nOpp = s.getPlayerBots(1 - myP, oppIdx);

    // Compute all 8 features (same as training)
    double scoreDiff = (double)(s.score(myP) - s.score(1 - myP));
    double botDiff   = (double)(nMy - nOpp);
    
    double myDist = 0, oppDist = 0;
    for (int i = 0; i < nMy; i++) {{
        int best = 9999; Point h = s.bots[myIdx[i]].body()[0];
        for (int a = 0; a < s.num_apples; a++) {{
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < best) best = d;
        }}
        myDist += (best < 9999 ? best : 0);
    }}
    for (int i = 0; i < nOpp; i++) {{
        int best = 9999; Point h = s.bots[oppIdx[i]].body()[0];
        for (int a = 0; a < s.num_apples; a++) {{
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < best) best = d;
        }}
        oppDist += (best < 9999 ? best : 0);
    }}
    double avgMyDist  = nMy  > 0 ? myDist  / nMy  : 0.0;
    double avgOppDist = nOpp > 0 ? oppDist / nOpp : 0.0;

    bool bodyGrid[MAX_DIM][MAX_DIM];
    memset(bodyGrid, 0, sizeof(bodyGrid));
    for (int i = 0; i < s.num_bots; i++) {{
        if (!s.bots[i].alive) continue;
        for (int j = 0; j < s.bots[i].body_size; j++) {{
            int bx = s.bots[i].body()[j].x, by = s.bots[i].body()[j].y;
            if (bx >= 0 && bx < W && by >= 0 && by < H) bodyGrid[by][bx] = true;
        }}
    }}
    double mySafety = 0, oppSafety = 0;
    for (int i = 0; i < nMy; i++) {{
        Point h = s.bots[myIdx[i]].body()[0]; int safe = 0;
        for (int d = 0; d < 4; d++) {{
            int nx = h.x + DX[d], ny = h.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !globalIsPlatform[ny][nx] && !bodyGrid[ny][nx]) safe++;
        }}
        mySafety += safe;
    }}
    for (int i = 0; i < nOpp; i++) {{
        Point h = s.bots[oppIdx[i]].body()[0]; int safe = 0;
        for (int d = 0; d < 4; d++) {{
            int nx = h.x + DX[d], ny = h.y + DY[d];
            if (nx >= 0 && nx < W && ny >= 0 && ny < H && !globalIsPlatform[ny][nx] && !bodyGrid[ny][nx]) safe++;
        }}
        oppSafety += safe;
    }}
    double avgMySafety  = nMy  > 0 ? mySafety  / nMy  : 0.0;
    double avgOppSafety = nOpp > 0 ? oppSafety / nOpp : 0.0;

    double myGround = 0, oppGround = 0;
    for (int i = 0; i < nMy; i++) {{
        Point h = s.bots[myIdx[i]].body()[0]; int by = h.y + 1;
        if (by < H && (globalIsPlatform[by][h.x] || bodyGrid[by][h.x])) myGround += 1.0;
        else if (by >= H) myGround += 0.5;
    }}
    for (int i = 0; i < nOpp; i++) {{
        Point h = s.bots[oppIdx[i]].body()[0]; int by = h.y + 1;
        if (by < H && (globalIsPlatform[by][h.x] || bodyGrid[by][h.x])) oppGround += 1.0;
        else if (by >= H) oppGround += 0.5;
    }}
    double avgMyGround  = nMy  > 0 ? myGround  / nMy  : 0.0;
    double avgOppGround = nOpp > 0 ? oppGround / nOpp : 0.0;

    double myMobility = 0, oppMobility = 0;
    for (int i = 0; i < nMy; i++) myMobility += floodFillCount(s.bots[myIdx[i]].body()[0], s);
    for (int i = 0; i < nOpp; i++) oppMobility += floodFillCount(s.bots[oppIdx[i]].body()[0], s);
    double avgMyMob  = nMy  > 0 ? myMobility  / nMy  : 0.0;
    double avgOppMob = nOpp > 0 ? oppMobility / nOpp : 0.0;

    double myOwned = 0, oppOwned = 0;
    for (int a = 0; a < s.num_apples; a++) {{
        int bmd = 9999, bod = 9999;
        for (int i = 0; i < nMy; i++) {{
            Point h = s.bots[myIdx[i]].body()[0];
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < bmd) bmd = d;
        }}
        for (int i = 0; i < nOpp; i++) {{
            Point h = s.bots[oppIdx[i]].body()[0];
            int d = abs(s.apples[a].x - h.x) + abs(s.apples[a].y - h.y);
            if (d < bod) bod = d;
        }}
        if (bmd < bod) myOwned += 1.0;
        else if (bod < bmd) oppOwned += 1.0;
    }}

    // Raw features
    float raw[{NUM_FEATURES}] = {{
        (float)scoreDiff, (float)botDiff, (float)avgMyDist, (float)avgOppDist,
        (float)(avgMySafety - avgOppSafety), (float)(avgMyGround - avgOppGround),
        (float)(avgMyMob - avgOppMob), (float)(myOwned - oppOwned)
    }};

    // Normalize
    float feat[{NUM_FEATURES}];
    for (int i = 0; i < {NUM_FEATURES}; i++)
        feat[i] = (raw[i] - NN_MEAN[i]) / NN_STD[i];

    // Layer 1: {NUM_FEATURES} → 32 (ReLU)
    float h1[32];
    for (int i = 0; i < 32; i++) {{
        h1[i] = NN_B1[i];
        for (int j = 0; j < {NUM_FEATURES}; j++) h1[i] += NN_W1[i][j] * feat[j];
        if (h1[i] < 0.0f) h1[i] = 0.0f;
    }}
    // Layer 2: 32 → 16 (ReLU)
    float h2[16];
    for (int i = 0; i < 16; i++) {{
        h2[i] = NN_B2[i];
        for (int j = 0; j < 32; j++) h2[i] += NN_W2[i][j] * h1[j];
        if (h2[i] < 0.0f) h2[i] = 0.0f;
    }}
    // Layer 3: 16 → 1 (Sigmoid)
    float out = NN_B3;
    for (int i = 0; i < 16; i++) out += NN_W3[i] * h2[i];

    return 1.0 / (1.0 + exp(-(double)out));
}}
"""

with open("nn_evaluate.cpp", "w") as f:
    f.write(cpp)
    f.write(eval_func)
    
print(f"\nSaved weights to: nn_weights.h")
print(f"Saved full evaluate function to: nn_evaluate.cpp")
print(f"\nTo use: copy the contents of nn_weights.h and nn_evaluate.cpp")
print(f"into bot.cpp, replacing the current evaluate() function + weight section.")