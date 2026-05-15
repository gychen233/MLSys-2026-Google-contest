#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std;

// ============================================================================
// [手动调整区] - 计算图与硬件环境参数配置
// ============================================================================

// 图结构配置
const int NUM_OPS = 200;               // 生成算子的总数量
const int REUSE_PROB = 70;            // (0-100) 尝试复用已有 Tensor 的概率（控制 DAG 复杂度）
const int PROB_POINTWISE = 60;        // (0-100) 生成 Pointwise 算子的概率（剩余为 MatMul）

// 维度大小范围 (2 的幂次指数范围)
const int MIN_DIM_EXP = 8;            // 最小维度 2^5 = 32
const int MAX_DIM_EXP = 13;           // 最大维度 2^13 = 8192

// Base Cost 范围
const int MIN_BASE_COST = 1000;
const int MAX_BASE_COST = 20000;

// 硬件参数随机范围
const int MIN_FAST_MEM = 10000000;    // ~10MB
const int MAX_FAST_MEM = 40000000;    // ~40MB
const int MIN_SLOW_BW = 16;          // 慢速内存带宽下限
const int MAX_SLOW_BW = 64;         // 慢速内存带宽上限

// Native Granularity 候选池 (通常是固定的几种硬件 tile 大小)
const vector<int> NATIVE_GRANULARITY_CHOICES_0 = {64, 128, 256};
const vector<int> NATIVE_GRANULARITY_CHOICES_1 = {64, 128, 256};

// ============================================================================

// 初始化全局随机数发生器
mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());

// 辅助函数：生成范围内的随机整数
int randInt(int min_val, int max_val) {
    uniform_int_distribution<> dist(min_val, max_val);
    return dist(gen);
}

// 辅助函数：生成 2 的幂次
int getRandomPowerOf2() {
    int exponent = randInt(MIN_DIM_EXP, MAX_DIM_EXP);
    return 1 << exponent;
}

struct Tensor {
    int id; // 原始生成的顺序 ID
    int w;
    int h;
};

struct Operation {
    int id;
    string type;
    vector<int> inputs;
    vector<int> outputs;
    int base_cost;
};

int main(int argc, char* argv[]) {
    vector<Tensor> tensors;
    vector<Operation> ops;

    // 1. 初始化根节点 Tensor
    tensors.push_back({0, getRandomPowerOf2(), getRandomPowerOf2()});

    // 2. 按拓扑序逐步生成 Operations
    for (int i = 0; i < NUM_OPS; ++i) {
        Operation op;
        op.id = i; // Node ID 保持顺序
        op.base_cost = randInt(MIN_BASE_COST, MAX_BASE_COST);

        if (randInt(1, 100) <= PROB_POINTWISE) {
            // 生成 Pointwise 算子
            op.type = "Pointwise";
            Tensor t1 = tensors[randInt(0, tensors.size() - 1)];
            op.inputs.push_back(t1.id);

            // 50% 概率为二元操作
            if (randInt(0, 1) == 0) {
                vector<int> valid_candidates;
                for (const auto& t : tensors) {
                    if (t.w == t1.w && t.h == t1.h) {
                        valid_candidates.push_back(t.id);
                    }
                }

                if (!valid_candidates.empty() && randInt(1, 100) <= REUSE_PROB) {
                    int t2_id = valid_candidates[randInt(0, valid_candidates.size() - 1)];
                    op.inputs.push_back(t2_id);
                } else {
                    Tensor t2 = {(int)tensors.size(), t1.w, t1.h};
                    tensors.push_back(t2);
                    op.inputs.push_back(t2.id);
                }
            }
            Tensor out = {(int)tensors.size(), t1.w, t1.h};
            tensors.push_back(out);
            op.outputs.push_back(out.id);

        } else {
            // 生成 MatMul 算子
            op.type = "MatMul";
            Tensor tL = tensors[randInt(0, tensors.size() - 1)];
            op.inputs.push_back(tL.id);
            
            int K = tL.w;
            int H = tL.h;
            
            vector<int> valid_candidates;
            for (const auto& t : tensors) {
                if (t.h == K) valid_candidates.push_back(t.id);
            }

            int W; 
            if (!valid_candidates.empty() && randInt(1, 100) <= REUSE_PROB) {
                int tR_id = valid_candidates[randInt(0, valid_candidates.size() - 1)];
                op.inputs.push_back(tR_id);
                W = tensors[tR_id].w; 
            } else {
                W = getRandomPowerOf2();
                Tensor tR = {(int)tensors.size(), W, K};
                tensors.push_back(tR);
                op.inputs.push_back(tR.id);
            }

            Tensor out = {(int)tensors.size(), W, H};
            tensors.push_back(out);
            op.outputs.push_back(out.id);
        }
        ops.push_back(op);
    }

    // ============================================================================
    // 3. 核心：打乱 Tensor 编号（双射映射）
    // ============================================================================
    int num_tensors = tensors.size();
    vector<int> id_map(num_tensors);
    iota(id_map.begin(), id_map.end(), 0); 
    shuffle(id_map.begin(), id_map.end(), gen); // id_map[old_id] = new_id

    // 重新排列 width 和 height 数组以适应新的 Tensor ID
    vector<int> final_widths(num_tensors);
    vector<int> final_heights(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        int new_id = id_map[i];
        final_widths[new_id] = tensors[i].w;
        final_heights[new_id] = tensors[i].h;
    }

    // 更新每个 Node 中的 input 和 output 编号
    for (auto& op : ops) {
        for (auto& in_id : op.inputs) in_id = id_map[in_id];
        for (auto& out_id : op.outputs) out_id = id_map[out_id];
    }

    // ============================================================================
    // 4. 序列化为 JSON 格式并写入文件
    // ============================================================================
    ofstream outFile(argv[1]);
    if (!outFile.is_open()) {
        cerr << "Fail!" << endl;
        return 1;
    }

    outFile << "{\n";
    
    auto writeArray = [&](const string& name, const vector<int>& arr) {
        outFile << "  \"" << name << "\": [";
        for (size_t i = 0; i < arr.size(); ++i) {
            outFile << arr[i] << (i + 1 == arr.size() ? "" : ", ");
        }
        outFile << "],\n";
    };

    writeArray("widths", final_widths);
    writeArray("heights", final_heights);

    outFile << "  \"inputs\": [\n";
    for (size_t i = 0; i < ops.size(); ++i) {
        outFile << "    [";
        for (size_t j = 0; j < ops[i].inputs.size(); ++j) {
            outFile << ops[i].inputs[j] << (j + 1 == ops[i].inputs.size() ? "" : ", ");
        }
        outFile << "]" << (i + 1 == ops.size() ? "\n" : ",\n");
    }
    outFile << "  ],\n";

    outFile << "  \"outputs\": [\n";
    for (size_t i = 0; i < ops.size(); ++i) {
        outFile << "    [";
        for (size_t j = 0; j < ops[i].outputs.size(); ++j) {
            outFile << ops[i].outputs[j] << (j + 1 == ops[i].outputs.size() ? "" : ", ");
        }
        outFile << "]" << (i + 1 == ops.size() ? "\n" : ",\n");
    }
    outFile << "  ],\n";

    outFile << "  \"base_costs\": [";
    for (size_t i = 0; i < ops.size(); ++i) {
        outFile << ops[i].base_cost << (i + 1 == ops.size() ? "" : ", ");
    }
    outFile << "],\n";

    outFile << "  \"op_types\": [";
    for (size_t i = 0; i < ops.size(); ++i) {
        outFile << "\"" << ops[i].type << "\"" << (i + 1 == ops.size() ? "" : ", ");
    }
    outFile << "],\n";

    // 随机生成硬件参数
    int rand_native_granularity_0 = NATIVE_GRANULARITY_CHOICES_0[randInt(0, NATIVE_GRANULARITY_CHOICES_0.size() - 1)];
    int rand_native_granularity_1 = NATIVE_GRANULARITY_CHOICES_1[randInt(0, NATIVE_GRANULARITY_CHOICES_1.size() - 1)];

    outFile << "  \"fast_memory_capacity\": " << randInt(MIN_FAST_MEM, MAX_FAST_MEM) << ",\n"; 
    outFile << "  \"slow_memory_bandwidth\": " << randInt(MIN_SLOW_BW, MAX_SLOW_BW) << ",\n";   
    outFile << "  \"native_granularity\": [" << rand_native_granularity_0 << ", " << rand_native_granularity_1 << "]\n";
    outFile << "}\n";

    outFile.close();
    printf("Success!\n");

    return 0;
}