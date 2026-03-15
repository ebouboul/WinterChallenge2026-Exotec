
// ==================== NEURAL NETWORK WEIGHTS (auto-generated) ====================
// Architecture: 8 → 32 → 16 → 1 (ReLU, ReLU, Sigmoid)
// Test Accuracy: 75.69%
// Features: score_diff, bot_diff, avg_my_dist, avg_opp_dist, safety_diff, ground_diff, mobility_diff, territory_diff

// Feature normalization constants
constexpr float NN_MEAN[8] = { 2.43205f,0.60181f,4.35339f,4.56149f,0.19353f,0.02815f,0.00000f,2.10660f };
constexpr float NN_STD[8] = { 3.43330f,0.80407f,1.90945f,2.16085f,0.79645f,0.41260f,1.00000f,5.47578f };

// Layer 1: 8 → 32
constexpr float NN_W1[32][8] = {
    {-1.13599f,0.06807f,-1.55406f,-0.10958f,-0.72728f,-0.00029f,0.02006f,1.56767f},    {-0.05648f,1.34281f,-0.42237f,0.15423f,-0.12720f,1.76616f,-0.26313f,-0.18696f},    {0.98016f,0.86823f,-1.96541f,-0.82212f,0.30220f,-0.48622f,0.14454f,1.04088f},    {-0.29367f,-1.94528f,-0.63218f,-0.21976f,0.24485f,-0.06080f,-0.23485f,-0.67000f},    {-1.02984f,0.71938f,-0.15076f,-0.02842f,0.03822f,-2.80093f,0.06861f,-0.15302f},    {-1.10428f,0.92419f,-0.03954f,1.18058f,-0.22065f,-1.46974f,0.06289f,-2.86617f},    {-0.60357f,-0.09424f,-0.69532f,0.07527f,-0.13790f,-2.99006f,0.21187f,-0.43139f},    {0.87633f,0.00334f,-0.24066f,0.34220f,0.08113f,-0.00328f,-0.25951f,-3.49858f},    {0.68323f,-0.01061f,3.41841f,0.19423f,0.05578f,-0.01475f,-0.28474f,-1.40003f},    {-0.00574f,0.44284f,-2.35338f,0.35016f,0.72055f,0.35554f,-0.16472f,-0.37102f},    {-0.84356f,0.63753f,-0.35392f,0.37863f,0.03338f,-2.11568f,0.09650f,-1.42467f},    {0.42315f,-0.59613f,1.06051f,0.69532f,0.06857f,-1.12171f,0.14128f,-1.95337f},    {0.98459f,0.91636f,0.10592f,-0.88059f,-0.25015f,0.14989f,-0.25721f,0.92326f},    {2.97017f,-2.31933f,-0.22067f,-0.16714f,0.20552f,-0.08883f,0.29681f,-0.14791f},    {-2.02979f,1.39935f,1.04860f,-0.24262f,0.05769f,-0.20516f,0.14530f,2.05701f},    {-0.59890f,2.13168f,-0.60965f,0.66593f,-0.26561f,0.13242f,-0.13171f,-0.29870f},    {0.11518f,-0.31944f,-1.06528f,-0.24693f,-1.09612f,-0.28865f,-0.08651f,0.11539f},    {1.52463f,-1.14338f,-0.05615f,-0.07348f,-0.22616f,-0.09594f,-0.35236f,2.06490f},    {1.24854f,-0.69880f,-1.20862f,0.55783f,-0.24834f,1.00927f,0.11904f,1.36705f},    {-0.49490f,1.27581f,0.40105f,-1.01468f,-0.27420f,0.04506f,0.25427f,-1.92845f},    {-0.06768f,0.53075f,0.16111f,-0.21432f,1.37076f,-0.02016f,0.29535f,0.62937f},    {0.24805f,0.25012f,0.71378f,-0.59357f,1.09810f,0.17104f,0.16150f,1.60905f},    {0.03713f,-0.64144f,-0.16253f,-2.19164f,0.17483f,-0.68357f,-0.19318f,-0.36654f},    {-1.61231f,1.17810f,0.88208f,-1.52132f,1.24127f,0.43126f,-0.17683f,0.27136f},    {0.06292f,-0.59090f,0.34854f,1.49468f,-0.24834f,-0.05987f,-0.12676f,-2.21660f},    {-0.07801f,-1.43100f,0.03883f,-0.77651f,0.11285f,-0.90408f,0.16512f,0.17767f},    {0.43959f,-0.28389f,0.53136f,0.26159f,0.01131f,-0.20369f,-0.21221f,2.69701f},    {-0.26693f,0.66295f,2.73749f,0.04850f,0.01417f,0.07792f,-0.18971f,-0.40248f},    {-2.16668f,1.70102f,-0.10544f,-1.28496f,0.08611f,0.15344f,-0.27970f,1.02325f},    {0.08160f,-1.64065f,2.02757f,-0.12330f,0.01182f,-0.06447f,-0.19155f,0.91776f},    {0.98122f,0.84233f,0.42010f,0.38972f,-0.02126f,0.28320f,-0.35084f,-1.82691f},    {-2.44345f,1.26915f,-1.18273f,-0.01293f,-0.03247f,-0.00221f,0.06743f,0.25199f}
};
constexpr float NN_B1[32] = { 0.27931f,-1.60975f,-1.23631f,-1.04564f,-0.27536f,-0.51548f,-2.16962f,0.72292f,0.52408f,-0.66703f,-0.13038f,0.76071f,0.62432f,-0.40434f,0.43306f,0.49893f,-0.08187f,-0.05645f,-0.15270f,-0.99652f,0.00629f,1.53509f,-0.87204f,-0.81647f,-0.65096f,-0.14740f,1.88916f,1.68509f,-0.09893f,1.08831f,1.12194f,-0.15608f };

// Layer 2: 32 → 16
constexpr float NN_W2[16][32] = {
    {0.40073f,-0.02773f,-0.48617f,-1.62338f,-1.08273f,-1.02100f,-1.15510f,1.56891f,0.92020f,-0.22032f,1.00441f,-0.03374f,1.03662f,0.59266f,-2.04021f,-0.40929f,-0.45723f,-0.72308f,-0.24871f,-1.18481f,-0.18764f,0.07986f,-0.44488f,0.42135f,-0.45968f,0.24707f,1.75082f,-1.78658f,-0.34287f,-0.21218f,-0.27905f,0.23536f},    {0.86063f,4.38100f,-1.54376f,-1.38354f,0.49596f,-0.32823f,0.73293f,-1.21711f,-0.60911f,1.62371f,-0.18159f,-0.35052f,0.72952f,0.83047f,-0.32651f,-5.14657f,-0.86255f,-2.10965f,-2.71011f,0.24571f,0.25126f,-0.99012f,1.15385f,-0.23283f,1.62317f,-0.39130f,-1.46811f,0.01721f,-1.57278f,0.21451f,-3.13190f,0.84411f},    {1.02064f,0.55461f,0.58059f,0.70486f,-1.45504f,-1.71262f,1.77684f,-0.52781f,-12.68035f,0.94871f,0.23698f,2.22987f,-0.36499f,-2.45213f,-0.15515f,0.00850f,-0.49086f,-0.54266f,-0.80294f,0.43366f,0.01142f,-0.56519f,0.06623f,-0.13829f,-1.42527f,-0.60099f,0.01271f,1.33797f,-0.48610f,-5.45890f,-0.71376f,-1.15406f},    {1.46807f,1.40661f,1.22370f,-0.03121f,-0.87231f,-0.38965f,1.99997f,-1.77634f,0.34173f,1.48169f,-1.08925f,2.35907f,-0.59015f,-1.55656f,-1.43017f,-0.67135f,-0.53913f,-1.68078f,-0.45845f,0.68241f,0.22673f,-0.89696f,0.86412f,-0.11938f,-0.98407f,-0.61829f,-0.20619f,0.55330f,-1.10085f,-1.82921f,-0.20866f,-0.53062f},    {1.54138f,0.04328f,-1.08005f,0.01745f,-0.76331f,-0.33123f,0.65993f,0.62677f,-0.96997f,1.05237f,1.42060f,0.37424f,0.72315f,-0.57261f,0.86935f,-0.43298f,-0.35363f,1.71340f,-0.93828f,0.34690f,-0.39729f,0.04802f,-2.11832f,1.93891f,0.21786f,1.28276f,-2.15047f,0.54922f,-2.47098f,-0.01682f,1.46989f,-0.53682f},    {0.26743f,0.39752f,-1.41516f,0.94465f,1.81509f,-0.65394f,0.05066f,0.32271f,1.39922f,0.29811f,0.14846f,-0.41796f,0.65592f,-2.14472f,-1.16677f,0.19919f,-0.86417f,-1.64504f,-0.50284f,-0.44988f,1.12961f,-0.34356f,1.29908f,0.11020f,-0.02004f,-0.44367f,0.46446f,0.37625f,0.43475f,0.37273f,-0.68912f,-0.54833f},    {1.61469f,-0.70857f,-1.06269f,0.24997f,0.22600f,0.59264f,0.35209f,0.68980f,-1.21982f,0.63431f,0.73275f,-0.08908f,0.85357f,0.28917f,0.28489f,-0.74004f,0.17059f,0.16242f,0.25050f,-1.66302f,0.30355f,-0.20860f,-0.84680f,1.18088f,-0.00835f,1.70372f,-1.12486f,1.48517f,-1.42644f,0.06594f,0.35107f,-1.33694f},    {1.05688f,-0.10754f,-0.65277f,-3.57325f,-3.10358f,-3.12376f,2.50198f,0.08169f,-0.50495f,0.48763f,1.59031f,1.56956f,0.50477f,2.20641f,0.19343f,-0.38370f,-1.11462f,0.63485f,-0.48395f,0.01393f,1.01775f,-1.18835f,1.27848f,-0.03548f,1.44752f,-2.30719f,0.23965f,0.38373f,-0.54611f,-0.86738f,0.47716f,-0.92647f},    {-0.72717f,0.36646f,0.36999f,0.09769f,0.26052f,-0.66133f,0.83003f,0.82680f,-0.75164f,1.28973f,-0.69157f,0.57072f,0.24902f,0.79019f,1.84947f,-1.21647f,0.11850f,-1.21890f,-1.14902f,1.35225f,-0.96098f,-1.58555f,-2.16985f,1.35859f,-0.33403f,0.68827f,-1.37185f,0.81869f,-1.80888f,-0.11044f,1.20625f,0.90797f},    {-0.84406f,3.33917f,-2.10095f,-0.82465f,-0.85592f,-0.19964f,-0.61890f,-2.33570f,-0.96716f,0.75151f,1.42590f,0.84087f,-11.77159f,-2.89662f,0.36434f,-0.43809f,1.24381f,-2.17317f,-0.88013f,2.06418f,1.39783f,-1.49681f,0.72086f,0.25021f,-0.21528f,0.48684f,-0.07159f,0.86127f,-0.34328f,-0.57544f,-1.88564f,0.36484f},    {0.71577f,1.46485f,-2.88418f,-1.09556f,-0.16649f,-1.22756f,0.34170f,-0.42339f,0.10621f,1.68839f,0.57437f,1.15625f,-0.02030f,-0.13372f,-2.09444f,-3.64923f,-0.42950f,1.31095f,-0.52444f,-1.05715f,0.83944f,-0.46366f,0.65016f,-1.34556f,-1.44962f,-0.54316f,-1.32555f,1.83832f,0.47263f,-1.64766f,-1.05605f,0.71903f},    {0.73622f,-0.41993f,-3.32791f,-0.76253f,0.24634f,-0.58929f,0.39388f,0.27575f,0.99292f,-0.80436f,0.96164f,-0.73211f,0.31434f,-1.07167f,-2.24342f,0.79215f,-0.63862f,-1.70453f,0.03282f,-0.17990f,0.64025f,0.13386f,0.38029f,0.98444f,1.34691f,-0.09138f,0.36039f,-1.14808f,0.45206f,1.67055f,0.55215f,0.47726f},    {-1.64206f,-1.56973f,0.09437f,-1.03758f,1.42985f,1.84430f,-2.13837f,1.00123f,-0.34714f,1.18297f,-0.86989f,-0.34171f,-0.20827f,-1.68663f,0.14711f,-1.02147f,1.06200f,0.84618f,1.27797f,1.11880f,-0.73599f,1.28944f,-0.36425f,-1.48865f,-1.73447f,1.59859f,-0.67503f,0.36331f,-0.45772f,-0.31430f,0.13405f,0.81253f},    {-0.61615f,0.59025f,-0.32558f,-0.79571f,1.24961f,-2.25566f,-0.32816f,-0.26139f,0.81414f,-0.84538f,-0.79024f,1.62461f,-0.44265f,-3.50172f,0.48104f,0.38715f,0.53693f,-0.91156f,-0.77466f,-0.80198f,0.31163f,0.95077f,-0.56734f,-1.33075f,-0.43005f,0.13360f,0.21981f,-0.54115f,0.82050f,-0.63752f,0.52238f,1.18931f},    {0.24548f,-0.23328f,0.18288f,-0.76817f,-0.05319f,0.11360f,1.01983f,-0.35207f,0.68094f,0.89607f,-0.95640f,0.23769f,0.85542f,1.01105f,-1.67765f,-0.01237f,-1.77111f,0.48966f,-0.47005f,0.31585f,-0.29685f,-0.28746f,-0.15333f,1.99442f,-0.57250f,1.45363f,-1.45442f,0.24523f,-1.53956f,-0.62664f,1.00313f,3.12360f},    {-0.32992f,0.17766f,-1.66348f,-0.47447f,0.60052f,-0.29961f,-8.71265f,-1.25108f,-0.84598f,0.88008f,1.34476f,-1.06979f,1.17968f,-3.18152f,-2.11291f,-0.62633f,-0.90174f,0.11522f,0.35840f,-1.08386f,1.50478f,-1.77829f,0.56396f,-0.20717f,0.02215f,-0.08018f,0.08928f,-28.87362f,0.73267f,0.09193f,0.28222f,0.67194f}
};
constexpr float NN_B2[16] = { 2.06551f,1.06793f,2.53407f,0.23008f,0.31314f,-3.23217f,-2.37951f,0.38728f,1.01380f,1.67991f,1.81333f,-0.12471f,-0.55222f,2.33422f,-2.28409f,1.60733f };

// Layer 3: 16 → 1
constexpr float NN_W3[16] = { 0.33440f,-1.03890f,-1.11955f,0.87414f,-0.44548f,-0.29786f,0.35465f,-0.28773f,0.25705f,-0.68813f,0.74543f,0.26439f,0.18579f,0.21586f,0.23079f,-1.13919f };
constexpr float NN_B3 = -0.52733f;
// =================================================================================

// ==================== EVALUATION (Neural Network) ====================

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

    // Compute all 8 features (same as training)
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

    // Raw features
    float raw[8] = {
        (float)scoreDiff, (float)botDiff, (float)avgMyDist, (float)avgOppDist,
        (float)(avgMySafety - avgOppSafety), (float)(avgMyGround - avgOppGround),
        (float)(avgMyMob - avgOppMob), (float)(myOwned - oppOwned)
    };

    // Normalize
    float feat[8];
    for (int i = 0; i < 8; i++)
        feat[i] = (raw[i] - NN_MEAN[i]) / NN_STD[i];

    // Layer 1: 8 → 32 (ReLU)
    float h1[32];
    for (int i = 0; i < 32; i++) {
        h1[i] = NN_B1[i];
        for (int j = 0; j < 8; j++) h1[i] += NN_W1[i][j] * feat[j];
        if (h1[i] < 0.0f) h1[i] = 0.0f;
    }
    // Layer 2: 32 → 16 (ReLU)
    float h2[16];
    for (int i = 0; i < 16; i++) {
        h2[i] = NN_B2[i];
        for (int j = 0; j < 32; j++) h2[i] += NN_W2[i][j] * h1[j];
        if (h2[i] < 0.0f) h2[i] = 0.0f;
    }
    // Layer 3: 16 → 1 (Sigmoid)
    float out = NN_B3;
    for (int i = 0; i < 16; i++) out += NN_W3[i] * h2[i];

    return 1.0 / (1.0 + exp(-(double)out));
}
