#pragma GCC optimize("Ofast")
#include <iostream>
#include <vector>
#include <deque>
#include <queue>
#include <set>
#include <string>
#include <random>
#include <chrono>
#include <fstream>

#include "json.hpp"
using json = nlohmann::json;

#define INF 1000000000000000
#define INF_LANTENCY 1000000000000000
#define init(a); memset(a, 0, sizeof(a));

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
#define ran(l, r) std::uniform_int_distribution<long long>(l, r)(rnd)
#define ran_real(l, r) std::uniform_real_distribution<double>(l, r)(rnd)

long long div_ceil(long long n, long long m) {
    return (n + m - 1) / m;
}

struct Tensor {
    long long height;
    long long width;
    long long input;
    std::vector<long long> outputs;
    bool is_cut;
    bool is_input;
    Tensor(): height(0), width(0), input(0), is_cut(0), is_input(0) {}
    Tensor(long long height, long long width) {
        this->height = height;
        this->width = width;
    }
    Tensor(long long height, long long width, long long input) {
        this->height = height;
        this->width = width;
        this->input = input;
        this->is_cut = 1;
        this->is_input = 0;
    }
};

struct Node {
    double cost;
    bool is_matmul;
    std::vector<long long> inputs;
    std::vector<long long> outputs;
};

void print(std::vector<std::vector<long long> >& subgraphs,
           std::vector<std::vector<long long> >& granularities,
           std::vector<std::vector<long long> >& tensors_to_retain,
           std::vector<std::vector<long long> >& traversal_orders) {
    for (long long i = 0; i < subgraphs.size(); i++) {
        for (long long j = 0; j < subgraphs[i].size(); j++) {
            printf("%lld ", subgraphs[i][j]);
        }
        printf("\n");
    }
    for (long long i = 0; i < granularities.size(); i++) {
        for (long long j = 0; j < granularities[i].size(); j++) {
            printf("%lld ", granularities[i][j]);
        }
        printf("\n");
    }
    for (long long i = 0; i < tensors_to_retain.size(); i++) {
        for (long long j = 0; j < tensors_to_retain[i].size(); j++) {
            printf("%lld ", tensors_to_retain[i][j]);
        }
        printf("\n");
    }
    for (long long i = 0; i < traversal_orders.size(); i++) {
        for (long long j = 0; j < traversal_orders[i].size(); j++) {
            printf("%lld ", traversal_orders[i][j]);
        }
        printf("\n");
    }
}

void printT(Tensor t) {
    printf("height: %lld, weight: %lld\n", t.height, t.width);
    printf("input: %lld\n", t.input);
    printf("output: ");
    for (long long i = 0 ; i < t.outputs.size(); i++) {
        printf("%lld ", t.outputs[i]);
    }
    printf("\n");
    printf("is_cut: %lld, is_input: %lld\n", t.is_cut, t.is_input);
    printf("\n");
}

class ScheduleEvaluator {
private:
    long long n, m;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
    long long fast_memory_capacity;
    double slow_memory_bandwidth;
    std::vector<long long> native_granularity;
    std::vector<std::vector<long long> > d_in, d_out;
    std::vector<long long> total_d_in, total_d_out;
    std::vector<long long> topo_id;
    long long n_layers;
    std::vector<std::vector<long long> > subgraphs;
    std::vector<std::vector<long long> > granularities;
    std::vector<std::vector<long long> > tensors_to_retain;
    std::vector<std::vector<long long> > traversal_orders;
    std::vector<bool> retain;
    std::vector<double> lantencies;

    std::vector<std::vector<long long> > h;
    std::vector<std::vector<long long> > w;
    std::vector<std::vector<bool> > is_h;
public:
    ScheduleEvaluator() : n(0), m(0), fast_memory_capacity(0), slow_memory_bandwidth(0.) {}

    ScheduleEvaluator (
        const std::vector<long long>& widths,
        const std::vector<long long>& heights,
        const std::vector<std::vector<long long> >& inputs,
        const std::vector<std::vector<long long> >& outputs,
        const std::vector<double>& base_costs,
        const std::vector<std::string>& op_types,
        const long long fast_memory_capacity,
        const double slow_memory_bandwidth,
        const std::vector<long long>& native_granularity
    ) {
        this->fast_memory_capacity = fast_memory_capacity;
        this->slow_memory_bandwidth = slow_memory_bandwidth;
        this->native_granularity = native_granularity;

        this->n = heights.size();
        assert(this->n == widths.size());
        for (long long i = 0; i < this->n; i++) {
            Tensor t;
            t.width = widths[i];
            t.height = heights[i];
            t.is_input = 1;
            this->tensors.push_back(t);

            h.push_back({0, 0});
            w.push_back({0, 0});
            is_h.push_back({false, false});
        }

        this->m = inputs.size();
        assert(this->m == outputs.size());
        assert(this->m == base_costs.size());
        assert(this->m == op_types.size());
        total_d_in.assign(n, 0);
        total_d_out.assign(n, 0);
        for (long long i = 0; i < m; i++) {
            Node nd;
            nd.inputs = inputs[i];
            nd.outputs = outputs[i];
            nd.cost = base_costs[i];
            if (op_types[i] == "MatMul") nd.is_matmul = 1;
            else nd.is_matmul = 0;
            nodes.push_back(nd);
            for (long long j = 0; j < nd.inputs.size(); j++) {
                tensors[nd.inputs[j]].outputs.push_back(i);
                total_d_in[nd.inputs[j]]++;
            }
            for (long long j = 0; j < nd.outputs.size(); j++) {
                tensors[nd.outputs[j]].input = i;
                tensors[nd.outputs[j]].is_input = 0;
                total_d_out[nd.outputs[j]]++;
            }
        }

        topo_id.resize(m);
        std::vector<long long> d(m, 0);
        for (long long i = 0; i < n; i++) {
            Tensor t = tensors[i];
            if (!t.is_input) {
                for (long long j = 0; j < t.outputs.size(); j++) {
                    d[t.outputs[j]]++;
                }
            }
        }
        std::queue<long long> q;
        for (long long i = 0; i < m; i++) {
            if (d[i] == 0) {
                q.push(i);
            }
        }
        long long cnt = 0;
        while(q.size()) {
            long long x = q.front();
            q.pop();
            topo_id[x] = ++cnt;
            Node nd = nodes[x];
            for (long long i = 0; i < nd.outputs.size(); i++) {
                Tensor t = tensors[nd.outputs[i]];
                for (long long j = 0; j < t.outputs.size(); j++) {
                    d[t.outputs[j]]--;
                    if (d[t.outputs[j]] == 0) {
                        q.push(t.outputs[j]);
                    }
                }
            }
        }
    }

    void get_necessity() {
        n_layers = subgraphs.size();
        d_in.resize(n_layers);
        d_out.resize(n_layers);
        for (long long i = 0; i < subgraphs.size(); i++) {
            d_in[i].assign(n, 0);
            d_out[i].assign(n, 0);
            for (long long j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (long long k = 0; k < nd.outputs.size(); k++) {
                    d_in[i][nd.outputs[0]]++;
                }
                for (long long k = 0; k < nd.inputs.size(); k++) {
                    d_out[i][nd.inputs[k]]++;
                }
            }
        }
    }

    long long fd_fa(long long x, std::vector<long long>& fa) {
        if (x != fa[x]) {
            fa[x] = fd_fa(fa[x], fa);
        }
        return fa[x];
    }

    bool get_subgraphs() {
        std::vector<long long> fa;
        for (long long i = 0; i < m; i++) {
            fa.push_back(i);
        }
        for (long long i = 0; i < n; i++) {
            Tensor t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (!t.is_cut) {
                long long x = fd_fa(t.input, fa);
                for (long long j = 0; j < t.outputs.size(); j++) {
                    fa[fd_fa(t.outputs[j], fa)] = x;
                }
            }
        }
        std::vector<long long> d(m, 0);
        std::vector<std::vector<long long> > out;
        out.resize(m);
        for (long long i = 0; i < n; i++) {
            Tensor t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (t.is_cut) {
                long long x = fd_fa(t.input, fa);
                for (long long j = 0; j < t.outputs.size(); j++) {
                    long long y = fd_fa(t.outputs[j], fa);
                    if (x != y) {
                        out[x].push_back(y);
                        d[y]++;
                    }
                }
            }
        }
        std::vector<long long> order;
        std::deque<long long> dq;
        for (long long i = 0; i < m; i++) {
            if (fa[i] == i) {
                if (d[i] == 0) {
                    dq.push_front(i);
                }
            }
        }
        while(dq.size()) {
            long long x = dq.front();
            dq.pop_front();
            order.push_back(x);
            bool flag = 0;
            for (long long i = 0; i < out[x].size(); i++) {
                d[out[x][i]]--;
                if (d[out[x][i]] == 0) {
                    if (!flag) {
                        dq.push_front(out[x][i]);
                        flag = 1;
                    } else {
                        dq.push_back(out[x][i]);
                    }
                }
            }
        }

        // bottle neck
        subgraphs.clear();
        for (long long i = 0; i < order.size(); i++) {
            std::vector<long long> subgraph;
            for (long long j = 0; j < m; j++) {
                if (fd_fa(j, fa) == order[i]) {
                    subgraph.push_back(j);
                }
            }
            // 由于有额外的 this 键字，所以要写成 lambda 表达式
            sort(subgraph.begin(), subgraph.end(), [this](long long x, long long y) {
                return this->topo_id[x] < this->topo_id[y];
            });
            subgraphs.push_back(subgraph);
        }

        // for (long long i = 0; i < subgraphs.size(); i++) {
        //     for (long long j = 0; j < subgraphs[i].size(); j++) {
        //         printf("%lld ", subgraphs[i][j]);
        //     }
        //     printf("\n");
        // }

        get_necessity();

        // Improve the inner order of subgraphs
        for (long long i = 0; i < subgraphs.size(); i++) {
            // if (d_out[i][nodes[subgraphs[i].back()].outputs[0]] != 0) {
            //     for (int j = 0; j < subgraphs[i].size(); j++) {
            //         printf("%d ", subgraphs[i][j]);
            //     }
            //     printf("\n");
            //     exit(0);
            // }
            for (long long j = 0; j < subgraphs[i].size() - 1; j++) {
                if (d_out[i][nodes[subgraphs[i][j]].outputs[0]] == 0) {
                    // printf("Multiple outputs!\n");
                    return 0;
                }
            }
        }

        // Check the legitimacy
        std::vector<bool> used_op(m, false), is_computed(n, false);
        for (long long i = 0; i < subgraphs.size(); i++) {
            for (long long j = 0; j < subgraphs[i].size(); j++) {
                used_op[subgraphs[i][j]] = true;
            }
        }
        for (long long i = 0; i < m; i++) {
            if (!used_op[i]) {
                // printf("!used_op[i]\n");
                return 0; // assert
            }
        }
        for (long long i = 0; i < n; i++) {
            is_computed[i] = true;
        }
        for (long long i = 0; i < m; i++) {
            for (long long j = 0; j < this->nodes[i].outputs.size(); j++) {
                is_computed[nodes[i].outputs[j]] = false;
            }
        }

        std::vector<bool> ephemeral_computed;
        for (long long i = 0; i < subgraphs.size(); i++) {
            ephemeral_computed.assign(n, false);
            for (long long j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (long long k = 0; k < nd.inputs.size(); k++) {
                    if (d_in[i][nd.inputs[k]] > 0) {
                        if (!ephemeral_computed[nd.inputs[k]]) {
                            // printf("!ephemeral_computed[nd.inputs[k]]\n");
                            return 0; // assert
                        }
                    }
                    else {
                        if (!is_computed[nd.inputs[k]]) {
                            // printf("!is_computed[nd.inputs[k]]\n");
                            return 0; // assert
                        }
                    }
                }
                for (long long k = 0; k < nd.outputs.size(); k++) {
                    ephemeral_computed[nd.outputs[k]] = true;
                }
            }
            for (long long j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (long long k = 0; k < nd.outputs.size(); k++) {
                    if (d_out[i][nodes[subgraphs[i][j]].outputs[k]] == 0) {
                        is_computed[nodes[subgraphs[i][j]].outputs[k]] = true;
                    }
                }
            }
        }
        return 1;
    }

    void upd_retain(long long layer) {
        tensors_to_retain[layer].clear();
        if (retain[layer]) {
            tensors_to_retain[layer].push_back(nodes[subgraphs[layer].back()].outputs[0]);
        }
    }

    void get_retain() {
        tensors_to_retain.resize(n_layers);
        for (long long layer = 0; layer < n_layers; layer++) {
            upd_retain(layer);
        }
    }

    long long get_point(long long n, long long m, long long x, long long y) {
        return x * m + y;
    }

    double upd_traversal_order(long long layer) {
        long long n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularities[layer][0];
        long long n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularities[layer][1];

        traversal_orders[layer].clear();
        
        if (n_h > 1 && n_w > 1) {
            double h_lantency, w_lantency;
            traversal_orders[layer].clear();
            for (long long i = 0; i < n_h / 2; i++) {
                for (long long j = 0; j < n_w; j++) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i, j));
                }
                for (long long j = n_w - 1; j >= 0; j--) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i + 1, j));
                }
            }
            h_lantency = get_lantency(layer);
            
            traversal_orders[layer].clear();
            for (long long j = 0; j < n_w / 2; j++) {
                for (long long i = 0; i < n_h; i++) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, i, 2 * j));
                }
                for (long long i = n_h - 1; i >= 0; i--) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, i, 2 * j + 1));
                }
            }
            w_lantency = get_lantency(layer);

            if (h_lantency < w_lantency) {
                traversal_orders[layer].clear();
                for (long long i = 0; i < n_h / 2; i++) {
                    for (long long j = 0; j < n_w; j++) {
                        traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i, j));
                    }
                    for (long long j = n_w - 1; j >= 0; j--) {
                        traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i + 1, j));
                    }
                }
            }
            return std::min(h_lantency, w_lantency);
        } else {
            return get_lantency(layer);
        }
    }

    long long cal_memory(std::vector<long long> granularity, long long layer) {
        long long n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularity[0];
        long long n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularity[1];
        long long n_k = nodes[subgraphs[layer].back()].is_matmul ? tensors[nodes[subgraphs[layer].back()].inputs[0]].width / granularity[2] : 1;
        for (long long i = 0; i < subgraphs[layer].size(); i++) {
            Node nd = nodes[subgraphs[layer][i]];
            for (long long j = 0; j < nd.inputs.size(); j++) {
                long long id = nd.inputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
            }
            for (long long j = 0; j < nd.outputs.size(); j++) {
                long long id = nd.outputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
            }
        }
        for (long long i = subgraphs[layer].size() - 1; i >= 0; i--) {
            Node nd = nodes[subgraphs[layer][i]];
            if (i == subgraphs[layer].size() - 1) {
                if (granularity[0] > native_granularity[0] || granularity[1] > native_granularity[1]) {
                    // printf("Exceed native_granularity!\n");
                    return INF_LANTENCY;
                }
                if (nd.is_matmul) {
                    h[nd.inputs[0]][0] = h[nd.inputs[0]][1] = granularity[0];
                    w[nd.inputs[0]][0] = w[nd.inputs[0]][1] = granularity[2];
                    h[nd.inputs[1]][0] = h[nd.inputs[1]][1] = granularity[2];
                    w[nd.inputs[1]][0] = w[nd.inputs[1]][1] = granularity[1];
                } else {
                    for (long long j = 0; j < nd.inputs.size(); j++) {
                        h[nd.inputs[j]][0] = h[nd.inputs[j]][1] = granularity[0];
                        w[nd.inputs[j]][0] = w[nd.inputs[j]][1] = granularity[1];
                    }
                }
            } else {
                if (nd.is_matmul) {
                    for (long long j = 0; j <= 1; j++) {
                        if (h[nd.outputs[0]][j] > h[nd.inputs[0]][0]) {
                            h[nd.inputs[0]][0] = h[nd.outputs[0]][j];
                            w[nd.inputs[0]][0] = tensors[nd.inputs[0]].width;
                        }
                        h[nd.inputs[0]][1] = h[nd.outputs[0]][j];
                        w[nd.inputs[0]][1] = tensors[nd.inputs[0]].width;


                        h[nd.inputs[1]][0] = tensors[nd.inputs[1]].height;
                        w[nd.inputs[1]][0] = w[nd.outputs[0]][j];
                        if (w[nd.outputs[0]][j] > w[nd.inputs[1]][1]) {
                            h[nd.inputs[1]][1] = tensors[nd.inputs[1]].height;
                            w[nd.inputs[1]][1] = w[nd.outputs[0]][j];
                        }
                    }
                } else {
                    for (long long j = 0; j <= 1; j++) {
                        for (long long k = 0; k < nd.inputs.size(); k++) {
                            if (h[nd.outputs[0]][j] > h[nd.inputs[k]][0]) {
                                h[nd.inputs[k]][0] = h[nd.outputs[0]][j];
                                w[nd.inputs[k]][0] = w[nd.outputs[0]][j];
                            }

                            if (w[nd.outputs[0]][j] > w[nd.inputs[k]][1]) {
                                h[nd.inputs[k]][1] = h[nd.outputs[0]][j];
                                w[nd.inputs[k]][1] = w[nd.outputs[0]][j];
                            }
                        }
                    }
                }
            }
        }

        long long retained_input = n;
        long long retain_memory = 0;
        if (layer > 0) {
            for (long long i = 0; i < tensors_to_retain[layer - 1].size(); i++) {
                retained_input = tensors_to_retain[layer - 1][i];
                retain_memory += tensors[tensors_to_retain[layer - 1][i]].height * tensors[tensors_to_retain[layer - 1][i]].width;
            }
        }
        for (long long i = 0; i < tensors_to_retain[layer].size(); i++) {
            retain_memory += tensors[tensors_to_retain[layer][i]].height * tensors[tensors_to_retain[layer][i]].width;
        }

        long long global_memory = 0;
        for (long long i = 0; i < subgraphs[layer].size(); i++) {
            Node nd = nodes[subgraphs[layer][i]];
            for (long long j = 0; j < nd.inputs.size(); j++) {
                long long id = nd.inputs[j];
                if (d_in[layer][id] == 0 && id != retained_input) {
                    if (h[id][0] > h[id][1] && w[id][0] < w[id][1]) {
                        global_memory += h[id][0] * w[id][0] + h[id][1] * w[id][1];
                    } else {
                        global_memory += std::max(h[id][0], h[id][1]) * std::max(w[id][0], w[id][1]);
                    }
                    h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
                }
            }
        }

        long long output_memory = 0;
        if (!retain[layer]) {
            output_memory += granularity[0] * granularity[1];
        }
        return retain_memory + output_memory + global_memory;
    }

    bool upd_granularity(long long layer) {
        std::vector<long long> granularity, best_granularity;
        std::vector<long long> best_traversal_order;
        double lantency, best_lantency = INF_LANTENCY;

        Node nd = nodes[subgraphs[layer].back()];
        long long H_lim = std::min(native_granularity[0], tensors[nd.outputs[0]].height);
        long long W_lim = std::min(native_granularity[1], tensors[nd.outputs[0]].width);

        long long K_lim = 1;
        if (nd.is_matmul) {
            K_lim = std::max(K_lim, tensors[nd.inputs[0]].width);
        }
        long long H, W;
        for (long long i = 0; i <= 2; i++) {
            for (long long j = 0; j <= i; j++) {
                H = H_lim >> j;
                W = W_lim >> (i - j);
                granularity = {H, W, 1};
                if (cal_memory(granularity, layer) <= fast_memory_capacity) {
                    for (long long K = K_lim; K >= 1; K >>= 1) {
                        granularity = {H, W, K};
                        if (cal_memory(granularity, layer) <= fast_memory_capacity) {
                            granularities[layer] = granularity;
                            lantency = upd_traversal_order(layer);
                            if (lantency < best_lantency) {
                                best_lantency = lantency;
                                best_granularity = granularity;
                                best_traversal_order = traversal_orders[layer];
                            }
                            break;
                        }
                    }
                }
            }
        }
        if (best_lantency < INF / 2) {
            lantencies[layer] = best_lantency;
            granularities[layer] = best_granularity;
            traversal_orders[layer] = best_traversal_order;
            return 1;
        } else {
            return 0;
        }
    }

    bool get_granularities() {
        granularities.resize(n_layers);
        for (long long layer = 0; layer < n_layers; layer++) {
            granularities[layer].clear();
            if(!upd_granularity(layer)) {
                return 0;
            }
        }
        return 1;
    }

    double get_lantency(long long layer) {
        long long n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularities[layer][0];
        long long n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularities[layer][1];
        long long n_k = nodes[subgraphs[layer].back()].is_matmul ? tensors[nodes[subgraphs[layer].back()].inputs[0]].width / granularities[layer][2] : 1;

        double lantency = 0.;
        double cal_lantency = 0.;
        for (long long i = 0; i < subgraphs[layer].size(); i++) {
            Node nd = nodes[subgraphs[layer][i]];
            for (long long j = 0; j < nd.inputs.size(); j++) {
                long long id = nd.inputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
                is_h[id][0] = is_h[id][1] = false;
            }
            for (long long j = 0; j < nd.outputs.size(); j++) {
                long long id = nd.outputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
                is_h[id][0] = is_h[id][1] = false;
            }
        }
        for (long long i = subgraphs[layer].size() - 1; i >= 0; i--) {
            Node nd = nodes[subgraphs[layer][i]];
            if (i == subgraphs[layer].size() - 1) {
                if (nd.is_matmul) {
                    h[nd.inputs[0]][0] = h[nd.inputs[0]][1] = granularities[layer][0];
                    w[nd.inputs[0]][0] = w[nd.inputs[0]][1] = granularities[layer][2];
                    is_h[nd.inputs[0]][0] = is_h[nd.inputs[0]][1] = true;
                    h[nd.inputs[1]][0] = h[nd.inputs[1]][1] = granularities[layer][2];
                    w[nd.inputs[1]][0] = w[nd.inputs[1]][1] = granularities[layer][1];
                    is_h[nd.inputs[1]][0] = is_h[nd.inputs[1]][1] = false;
                    cal_lantency += (double)nd.cost / n_k;
                } else {
                    for (long long j = 0; j < nd.inputs.size(); j++) {
                        h[nd.inputs[j]][0] = h[nd.inputs[j]][1] = granularities[layer][0];
                        w[nd.inputs[j]][0] = w[nd.inputs[j]][1] = granularities[layer][1];
                    }
                    cal_lantency += (double)nd.cost / n_k;
                }
            } else {
                if (h[nd.outputs[0]][0] > h[nd.outputs[0]][1] && w[nd.outputs[0]][0] < w[nd.outputs[0]][1]) {
                    cal_lantency += (double)nd.cost / n_k * 2;
                } else {
                    cal_lantency += (double)nd.cost / n_k;
                }
                if (nd.is_matmul) {
                    for (long long j = 0; j <= 1; j++) {
                        if (h[nd.outputs[0]][j] > h[nd.inputs[0]][0]) {
                            h[nd.inputs[0]][0] = h[nd.outputs[0]][j];
                            w[nd.inputs[0]][0] = tensors[nd.inputs[0]].width;
                            is_h[nd.inputs[0]][0] = is_h[nd.outputs[0]][j];
                        }
                        h[nd.inputs[0]][1] = h[nd.outputs[0]][j];
                        w[nd.inputs[0]][1] = tensors[nd.inputs[0]].width;
                        is_h[nd.inputs[0]][1] = is_h[nd.outputs[0]][j];

                        h[nd.inputs[1]][0] = tensors[nd.inputs[1]].height;
                        w[nd.inputs[1]][0] = w[nd.outputs[0]][j];
                        is_h[nd.inputs[1]][0] = is_h[nd.outputs[0]][j];
                        if (w[nd.outputs[0]][j] > w[nd.inputs[1]][1]) {
                            h[nd.inputs[1]][1] = tensors[nd.inputs[1]].height;
                            w[nd.inputs[1]][1] = w[nd.outputs[0]][j];
                            is_h[nd.inputs[1]][1] = is_h[nd.outputs[0]][j];
                        }
                    }
                } else {
                    for (long long j = 0; j <= 1; j++) {
                        for (long long k = 0; k < nd.inputs.size(); k++) {
                            if (h[nd.outputs[0]][j] > h[nd.inputs[k]][0]) {
                                h[nd.inputs[k]][0] = h[nd.outputs[0]][j];
                                w[nd.inputs[k]][0] = w[nd.outputs[0]][j];
                                is_h[nd.inputs[k]][0] = is_h[nd.outputs[0]][j];
                            }

                            if (w[nd.outputs[0]][j] > w[nd.inputs[k]][1]) {
                                h[nd.inputs[k]][1] = h[nd.outputs[0]][j];
                                w[nd.inputs[k]][1] = w[nd.outputs[0]][j];
                                is_h[nd.inputs[k]][1] = is_h[nd.outputs[0]][j];
                            }
                        }
                    }
                }
            }
        }

        double global_memory = 0.;
        double part_memory[2] = {0., 0.};
        double k_memory = 0., full_memory = 0.;
        double part_k_memory[2] = {0., 0.};

        long long retained_input = n;
        if (layer > 0 && retain[layer - 1]) {
            retained_input = nodes[subgraphs[layer - 1].back()].outputs[0];
        }

        for (long long i = 0; i < subgraphs[layer].size(); i++) {
            Node nd = nodes[subgraphs[layer][i]];
            for (long long j = 0; j < nd.inputs.size(); j++) {
                long long id = nd.inputs[j];
                if (d_in[layer][id] == 0  && id != retained_input) {
                    if (h[id][0] > h[id][1] && w[id][0] < w[id][1]) {
                        global_memory += h[id][0] * w[id][0] + h[id][1] * w[id][1];
                    } else {
                        global_memory += std::max(h[id][0], h[id][1]) * std::max(w[id][0], w[id][1]);
                    }
                    if ((h[id][0] == tensors[id].height && w[id][0] == tensors[id].width)
                        || (h[id][1] == tensors[id].height && w[id][1] == tensors[id].width)) {
                        full_memory += h[id][0] * w[id][0];
                    } else {
                        if (h[id][0] == h[id][1] && w[id][0] == w[id][1]) {
                            part_memory[is_h[id][0]] += h[id][0] * w[id][0];
                            if ((is_h[id][0] && w[id][0] == tensors[id].width) || (!is_h[id][0] && h[id][0] == tensors[id].height)) {
                                part_k_memory[is_h[id][0]] += h[id][0] * w[id][0];
                            }
                        } else {
                            if (h[id][0] > h[id][1]) {
                                part_memory[is_h[id][0]] += h[id][0] * w[id][0];
                                if ((is_h[id][0] && w[id][0] == tensors[id].width) || (!is_h[id][0] && h[id][0] == tensors[id].height)) {
                                    part_k_memory[is_h[id][0]] += h[id][0] * w[id][0];
                                }
                            }
                            if (w[id][0] < w[id][1]) {
                                part_memory[is_h[id][1]] += h[id][1] * w[id][1];
                                if ((is_h[id][0] && w[id][0] == tensors[id].width) || (!is_h[id][0] && h[id][0] == tensors[id].height)) {
                                    part_k_memory[is_h[id][0]] += h[id][1] * w[id][1];
                                }
                            }
                        }
                    }
                }
            }
        }

        bool output_is_retained = retain[layer];
        for (long long h = 0; h < n_h; h++) {
            for (long long w = 0; w < n_w; w++) {
                bool link_h, link_w;
                if (traversal_orders[layer].size() == 0) {
                    link_h = 0;
                    link_w = (w != 0);
                } else {
                    if (h == 0 && w == 0) {
                        link_h = link_w = 0;
                    } else {
                        long long pre_h = traversal_orders[layer][h * n_w + w - 1] / n_w;
                        long long pre_w = traversal_orders[layer][h * n_w + w - 1] % n_w;
                        long long now_h = traversal_orders[layer][h * n_w + w] / n_w;
                        long long now_w = traversal_orders[layer][h * n_w + w] % n_w;
                        link_h = (pre_h == now_h);
                        link_w = (pre_w == now_w);
                    }
                }
                if (n_k <= 2) {
                    for (long long k = 0; k < n_k; k++) {
                        double turn_memory = global_memory;
                        if (!(h == 0 && w == 0 && k == 0)) {
                            turn_memory -= full_memory;
                            if (nodes[subgraphs[layer].back()].is_matmul) {
                                if (n_k > 1) {
                                    if (k == 0) {
                                        if (link_h) {
                                            turn_memory -= part_k_memory[0];
                                        }
                                        if (link_w) {
                                            turn_memory -= part_k_memory[1];
                                        }
                                    } else {
                                        turn_memory -= part_k_memory[0];
                                        turn_memory -= part_k_memory[1];
                                    }
                                } else {
                                    if (link_h) {
                                        turn_memory -= part_memory[0];
                                    }
                                    if (link_w) {
                                        turn_memory -= part_memory[1];
                                    }
                                }
                            }
                        }
                        if (k == n_k - 1 && !output_is_retained) {
                            turn_memory += granularities[layer][0] * granularities[layer][1];
                        }
                        if (h == n_h - 1 && w == n_w - 1 && k == n_k - 1) {
                            long long id = nodes[subgraphs[layer].back()].outputs[0];
                            if (total_d_out[id] == 0) {
                                if (output_is_retained) {
                                    turn_memory += tensors[id].height * tensors[id].width;
                                }
                            }
                        }
                        lantency += std::max(turn_memory / slow_memory_bandwidth, cal_lantency);
                    }
                } else {
                    double turn_memory = global_memory;
                    if (!(h == 0 && w == 0)) {
                        turn_memory -= full_memory;
                        if (nodes[subgraphs[layer].back()].is_matmul) {
                            if (link_h) {
                                turn_memory -= part_k_memory[0];
                            }
                            if (link_w) {
                                turn_memory -= part_k_memory[1];
                            }
                        }
                    }
                    lantency += std::max(turn_memory / slow_memory_bandwidth, cal_lantency);

                    turn_memory = global_memory;
                    turn_memory -= full_memory;
                    if (nodes[subgraphs[layer].back()].is_matmul) {
                        turn_memory -= part_k_memory[0];
                        turn_memory -= part_k_memory[1];
                    }
                    lantency += (n_k - 2) * std::max(turn_memory / slow_memory_bandwidth, cal_lantency);
                    
                    if (!output_is_retained) {
                        turn_memory += granularities[layer][0] * granularities[layer][1];
                    }
                    if (h == n_h - 1 && w == n_w - 1) {
                        long long id = nodes[subgraphs[layer].back()].outputs[0];
                        if (total_d_out[id] == 0) {
                            if (output_is_retained) {
                                turn_memory += tensors[id].height * tensors[id].width;
                            }
                        }
                    }
                    lantency += std::max(turn_memory / slow_memory_bandwidth, cal_lantency);
                }
            }
        }
        return lantency;
    }

    void get_lantencies() {
        lantencies.resize(n_layers);
        for (long long layer = 0; layer < n_layers; layer++) {
            lantencies[layer] = get_lantency(layer);
        }
    }

    double get_total_lantency() {
        double total_lantency = 0.;
        for (long long layer = 0; layer < n_layers; layer++) {
            total_lantency += lantencies[layer];
        }
        return total_lantency;
    }

    std::tuple<std::vector<Tensor>, std::vector<bool>, double> sa_solution() {
        double global_best_lantency, current_best_lantency;
        std::vector<Tensor> global_best_state;
        std::vector<bool> global_best_retain;
        for (long long i = 0; i < n; i++) {
            tensors[i].is_cut = 1;
        }
        
        get_subgraphs();
        retain.assign(n_layers, false);
        traversal_orders.resize(n_layers);
        lantencies.resize(n_layers);

        get_retain();
        get_granularities();
        get_lantencies();
        current_best_lantency = get_total_lantency();
        global_best_lantency = current_best_lantency;
        // printf("%lf\n", global_best_lantency);

        global_best_retain.assign(n_layers, false);
        global_best_state.resize(n);
        global_best_state = tensors;

        double init_temp_0 = 90000.0 * n;
        double min_temp_0 = 0.1;
        double cooling_rate_0 = 0.99;
        for (double temp_0 = init_temp_0; temp_0 >= min_temp_0;) {
            long long x = ran(0, n - 1);
            tensors[x].is_cut ^= 1;
            
            if (!get_subgraphs()) { tensors[x].is_cut ^= 1; }
            else {
                retain.assign(n_layers, false);
                traversal_orders.resize(n_layers);
                lantencies.resize(n_layers);

                get_retain();
                if (!get_granularities()) { tensors[x].is_cut ^= 1;}
                else {
                    temp_0 *= cooling_rate_0;

                    get_lantencies();
                    double current_lantency = get_total_lantency();
                    double best_lantency = current_lantency;
                    std::vector<bool> best_retain;
                    best_retain.assign(n_layers, false);

                    double init_temp_1 = 3400.0 * n;
                    double min_temp_1 = 0.1;
                    double cooling_rate_1 = 0.8;
                    for (double temp_1 = init_temp_1; temp_1 >= min_temp_1; temp_1 *= cooling_rate_1) {
                        if (n_layers > 1) {
                            long long layer = ran(0, n_layers - 2);
                            retain[layer] = !retain[layer];
                            upd_retain(layer);

                            double temp_lantency[2];
                            std::vector<long long> temp_granularty[2];
                            std::vector<long long> temp_traversal_order[2];

                            temp_lantency[0] = lantencies[layer];
                            temp_granularty[0] = granularities[layer];
                            temp_traversal_order[0] = traversal_orders[layer];

                            temp_lantency[1] = lantencies[layer + 1];
                            temp_granularty[1] = granularities[layer + 1];
                            temp_traversal_order[1] = traversal_orders[layer + 1];

                            if (!upd_granularity(layer) || !upd_granularity(layer + 1)) {
                                retain[layer] = !retain[layer];
                                upd_retain(layer);

                                lantencies[layer] = temp_lantency[0];
                                granularities[layer] = temp_granularty[0];
                                traversal_orders[layer] = temp_traversal_order[0];

                                lantencies[layer + 1] = temp_lantency[1];
                                granularities[layer + 1] = temp_granularty[1];
                                traversal_orders[layer + 1] = temp_traversal_order[1];

                                continue;
                            }

                            double new_lantency = current_lantency;
                            new_lantency += -temp_lantency[0] + lantencies[layer];
                            new_lantency += -temp_lantency[1] + lantencies[layer + 1];
                            
                            if (current_lantency > new_lantency || ran_real(0, 1) < std::exp((current_lantency - new_lantency) / temp_1)) {
                            
                                current_lantency = new_lantency;
                                
                                if (current_lantency < best_lantency) {
                                    best_retain = retain;
                                    best_lantency = current_lantency;
                                }
                            } else {
                                retain[layer] = !retain[layer];
                                upd_retain(layer);

                                lantencies[layer] = temp_lantency[0];
                                granularities[layer] = temp_granularty[0];
                                traversal_orders[layer] = temp_traversal_order[0];

                                lantencies[layer + 1] = temp_lantency[1];
                                granularities[layer + 1] = temp_granularty[1];
                                traversal_orders[layer + 1] = temp_traversal_order[1];
                            }
                        }
                    }
                    // print(subgraphs,
                    //       granularities,
                    //       tensors_to_retain,
                    //       traversal_orders);
                    // printf("%lf\n", best_lantency);
                    if (current_best_lantency > best_lantency || ran_real(0, 1) < std::exp((current_best_lantency - best_lantency) / temp_0)) {
                        if (best_lantency < global_best_lantency) {
                            global_best_lantency = best_lantency;
                            global_best_state = tensors;
                            global_best_retain = best_retain;
                        }
                        current_best_lantency = best_lantency;
                    } else {
                        tensors[x].is_cut ^= 1;
                    }
                }
            }
        }
        printf("%lf\n", global_best_lantency);
        return {global_best_state, global_best_retain, global_best_lantency};
    }

    void ini(std::vector<std::vector<long long> > subgraphs,
             std::vector<std::vector<long long> > granularities,
             std::vector<std::vector<long long> > tensors_to_retain,
             std::vector<std::vector<long long> > traversal_orders) {
        this->subgraphs = subgraphs;
        get_necessity();
        this->granularities = granularities;
        this->tensors_to_retain = tensors_to_retain;
        retain.resize(n_layers);
        for (int layer = 0; layer < n_layers; layer++) {
            if (tensors_to_retain[layer].empty()) {
                retain[layer] = false;
            } else {
                retain[layer] = true;
            }
        }
        this->traversal_orders = traversal_orders;  
    }

    std::vector<double> evaluate(std::vector<std::vector<long long> > subgraphs,
                                std::vector<std::vector<long long> > granularities,
                                std::vector<std::vector<long long> > tensors_to_retain,
                                std::vector<std::vector<long long> > traversal_orders) {
        ini(subgraphs,
            granularities,
            tensors_to_retain,
            traversal_orders);
        get_lantencies();
        return lantencies;
    }

    std::tuple<std::vector<std::vector<long long> >,
               std::vector<std::vector<long long> >,
               std::vector<std::vector<long long> >,
               std::vector<std::vector<long long> >,
               std::vector<double> > final_solution(std::vector<Tensor>& best_state,
                                                   std::vector<bool>& best_retain) {
        tensors = best_state;
        retain = best_retain;
        
        get_subgraphs();
        traversal_orders.resize(n_layers);
        get_retain();
        get_granularities();
        get_lantencies();
        return {subgraphs, granularities, tensors_to_retain, traversal_orders, lantencies};
    }

    bool check_OOM(std::vector<std::vector<long long> > subgraphs,
                   std::vector<std::vector<long long> > granularities,
                   std::vector<std::vector<long long> > tensors_to_retain,
                   std::vector<std::vector<long long> > traversal_orders) {
        ini(subgraphs,
            granularities,
            tensors_to_retain,
            traversal_orders);
        for (long long layer = 0; layer < n_layers; layer++) {
            if (cal_memory(granularities[layer], layer) > fast_memory_capacity) {
                return 0;
            }
        }
        return 1;
    }
};

void solve(const std::string file_in_name, const std::string file_out_name) {
    std::ifstream file_in(file_in_name);
    json j_in;
    file_in >> j_in; 

    std::vector<long long> widths = j_in["widths"];
    std::vector<long long> heights = j_in["heights"];
    std::vector<std::vector<long long> > inputs = j_in["inputs"];
    std::vector<std::vector<long long> > outputs = j_in["outputs"];
    std::vector<double> base_costs = j_in["base_costs"];
    std::vector<std::string> op_types = j_in["op_types"];
    double fast_memory_capacity = j_in["fast_memory_capacity"];
    double slow_memory_bandwidth = j_in["slow_memory_bandwidth"];
    std::vector<long long> native_granularity = j_in["native_granularity"];

    auto start = std::chrono::high_resolution_clock::now();
    ScheduleEvaluator myEvaluater(
        widths,
        heights,
        inputs,
        outputs,
        base_costs,
        op_types,
        fast_memory_capacity,
        slow_memory_bandwidth,
        native_granularity
    );

    std::vector<Tensor> best_state;
    std::vector<bool> best_retain;
    double best_lantency = INF_LANTENCY;

    for (long long t = 0; t < 3; t++) {
        auto [current_state, current_retain, current_lantency] = 
                                    myEvaluater.sa_solution();
        if (current_lantency < best_lantency) {
            best_state = current_state;
            best_retain = current_retain;
            best_lantency = current_lantency;
        }
    }

    printf("best lantency: %lf\n", best_lantency);

    auto [subgraphs, granularities, tensors_to_retain, traversal_orders, lantencies] =
            myEvaluater.final_solution(best_state, best_retain);
    // print(subgraphs, granularities, tensors_to_retain, traversal_orders);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "time:" << duration.count() << " ms" << std::endl;

    nlohmann::ordered_json j_out;
    j_out["subgraphs"] = subgraphs;
    j_out["granularities"] = granularities;
    j_out["tensors_to_retain"] = tensors_to_retain;
    for (long long i = 0; i < traversal_orders.size(); i++) {
        j_out["traversal_orders"].push_back(
            traversal_orders[i].empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(traversal_orders[i])
        );
    }
    j_out["subgraph_latencies"] = lantencies;
    std::ofstream file_out(file_out_name);
    file_out << j_out.dump(4);
    file_out.close();
}

void check_out(const std::string file_in_name, const std::string file_out_name, bool mode=0) {
    std::ifstream file_in(file_in_name);
    if (!file_in.is_open()) {
        std::cerr << "Can't open input.json!" << std::endl;
        return;
    }
    json j_in;
    file_in >> j_in; 

    std::vector<long long> widths = j_in["widths"];
    std::vector<long long> heights = j_in["heights"];
    std::vector<std::vector<long long> > inputs = j_in["inputs"];
    std::vector<std::vector<long long> > outputs = j_in["outputs"];
    std::vector<double> base_costs = j_in["base_costs"];
    std::vector<std::string> op_types = j_in["op_types"];
    double fast_memory_capacity = j_in["fast_memory_capacity"];
    double slow_memory_bandwidth = j_in["slow_memory_bandwidth"];
    std::vector<long long> native_granularity = j_in["native_granularity"];

    ScheduleEvaluator myEvaluater(
        widths,
        heights,
        inputs,
        outputs,
        base_costs,
        op_types,
        fast_memory_capacity,
        slow_memory_bandwidth,
        native_granularity
    );

    std::ifstream file_out(file_out_name);
    if (!file_out.is_open()) {
        std::cerr << "Can't open output.json!" << std::endl;
        return;
    }

    json j_out;
    file_out >> j_out;

    std::vector<std::vector<long long> > subgraphs = j_out["subgraphs"];
    std::vector<std::vector<long long> > granularities = j_out["granularities"];
    std::vector<std::vector<long long> > tensors_to_retain = j_out["tensors_to_retain"];
    std::vector<std::vector<long long> > traversal_orders;
    for (auto& item: j_out["traversal_orders"]) {
        if (item.is_null()) {
            traversal_orders.push_back({}); 
        } else {
            traversal_orders.push_back(item.get<std::vector<long long> >());
        }
    }

    if (!myEvaluater.check_OOM(subgraphs, granularities, tensors_to_retain, traversal_orders)) {
        printf("OOM!\n");
        return;
    }

    std::vector<double> lantencies = myEvaluater.evaluate(subgraphs, granularities, tensors_to_retain, traversal_orders);

    if (!mode) {
        std::vector<double> subgraph_latencies = j_out["subgraph_latencies"];
        for (long long i = 0; i < lantencies.size(); i++) {
            if (subgraph_latencies[i] != lantencies[i]) {
                printf("Lantency of subgraph %lld is wrong!\n", i);
            }
        }
        printf("Lantencies is right!\n");
    } else {
        for (long long i = 0; i < lantencies.size(); i++) {
            printf("%lf\n", lantencies[i]);
        }
        printf("%lf\n", myEvaluater.get_total_lantency());
    }
}

int main(int argc, char* argv[]) {
    // std::cout << "case: " << argv[1] << std::endl;
    // solve(argv[1], argv[2]);

    // solve("input.json", "output.json");

    check_out("input.json", "output.json", 1);
    return 0;
}