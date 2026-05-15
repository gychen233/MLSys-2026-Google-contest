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

#define INF_LANTENCY 1000000000000000
#define init(a); memset(a, 0, sizeof(a));

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
#define ran(l, r) std::uniform_int_distribution<int>(l, r)(rnd)
#define ran_real(l, r) std::uniform_real_distribution<double>(l, r)(rnd)

std::string INPUT_PATH;
std::string OUTPUT_PATH;

int div_ceil(int n, int m) {
    return (n + m - 1) / m;
}

struct Tensor {
    int height;
    int width;
    int input;
    std::vector<int> outputs;
    bool is_input;
    Tensor() : height(0), width(0), input(0), is_input(0) {}
    Tensor(int height, int width) {
        this->height = height;
        this->width = width;
    }
    Tensor(int height, int width, int input) {
        this->height = height;
        this->width = width;
        this->input = input;
        this->is_input = 0;
    }
};

struct Node {
    double cost;
    bool is_matmul;
    std::vector<int> inputs;
    std::vector<int> outputs;
};

void print(std::vector<std::vector<int> >& subgraphs,
    std::vector<std::vector<int> >& granularities,
    std::vector<std::vector<int> >& tensors_to_retain,
    std::vector<std::vector<int> >& traversal_orders) {
    for (int i = 0; i < subgraphs.size(); i++) {
        for (int j = 0; j < subgraphs[i].size(); j++) {
            printf("%d ", subgraphs[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < granularities.size(); i++) {
        for (int j = 0; j < granularities[i].size(); j++) {
            printf("%d ", granularities[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < tensors_to_retain.size(); i++) {
        for (int j = 0; j < tensors_to_retain[i].size(); j++) {
            printf("%d ", tensors_to_retain[i][j]);
        }
        printf("\n");
    }
    for (int i = 0; i < traversal_orders.size(); i++) {
        for (int j = 0; j < traversal_orders[i].size(); j++) {
            printf("%d ", traversal_orders[i][j]);
        }
        printf("\n");
    }
}

class ScheduleEvaluator {
private:
    int n, m;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
    long long fast_memory_capacity;
    double slow_memory_bandwidth;
    std::vector<int> native_granularity;
    std::vector<std::vector<int> > d_in, d_out;
    std::vector<int> total_d_in, total_d_out;
    std::vector<int> topo_seq;
    int n_layers;
    std::vector<bool> is_cut;
    std::vector<std::vector<int> > subgraphs;
    std::vector<std::vector<int> > granularities;
    std::vector<std::vector<int> > tensors_to_retain;
    std::vector<std::vector<int> > traversal_orders;
    std::vector<bool> retain;
    std::vector<double> lantencies;
    std::map<std::vector<bool>, double> memo;

    std::vector<std::vector<int> > h;
    std::vector<std::vector<int> > w;
    std::vector<std::vector<bool> > is_h;

    double best_lantency;
    std::vector<bool> best_cut;

public:
    ScheduleEvaluator() : n(0), m(0), fast_memory_capacity(0), slow_memory_bandwidth(0.) {}

    ScheduleEvaluator(
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        const std::vector<std::vector<int> >& inputs,
        const std::vector<std::vector<int> >& outputs,
        const std::vector<double>& base_costs,
        const std::vector<std::string>& op_types,
        const long long fast_memory_capacity,
        const double slow_memory_bandwidth,
        const std::vector<int>& native_granularity
    ) {
        this->fast_memory_capacity = fast_memory_capacity;
        this->slow_memory_bandwidth = slow_memory_bandwidth;
        this->native_granularity = native_granularity;

        this->n = heights.size();
        assert(this->n == widths.size());
        for (int i = 0; i < this->n; i++) {
            Tensor t;
            t.width = widths[i];
            t.height = heights[i];
            t.is_input = 1;
            this->tensors.push_back(t);

            h.push_back({ 0, 0 });
            w.push_back({ 0, 0 });
            is_h.push_back({ false, false });
        }

        this->m = inputs.size();
        assert(this->m == outputs.size());
        assert(this->m == base_costs.size());
        assert(this->m == op_types.size());
        total_d_in.assign(n, 0);
        total_d_out.assign(n, 0);
        for (int i = 0; i < m; i++) {
            Node nd;
            nd.inputs = inputs[i];
            nd.outputs = outputs[i];
            nd.cost = base_costs[i];
            if (op_types[i] == "MatMul") nd.is_matmul = 1;
            else nd.is_matmul = 0;
            nodes.push_back(nd);
            for (int j = 0; j < nd.inputs.size(); j++) {
                tensors[nd.inputs[j]].outputs.push_back(i);
                total_d_out[nd.inputs[j]]++;
            }
            for (int j = 0; j < nd.outputs.size(); j++) {
                tensors[nd.outputs[j]].input = i;
                tensors[nd.outputs[j]].is_input = 0;
                total_d_in[nd.outputs[j]]++;
            }
        }

        std::vector<int> d(m, 0);
        for (int i = 0; i < n; i++) {
            const Tensor& t = tensors[i];
            if (!t.is_input) {
                for (int j = 0; j < t.outputs.size(); j++) {
                    d[t.outputs[j]]++;
                }
            }
        }
        std::queue<int> q;
        for (int i = 0; i < m; i++) {
            if (d[i] == 0) {
                q.push(i);
            }
        }
        while (q.size()) {
            int x = q.front();
            q.pop();
            topo_seq.push_back(x);
            const Node& nd = nodes[x];
            for (int i = 0; i < nd.outputs.size(); i++) {
                const Tensor& t = tensors[nd.outputs[i]];
                for (int j = 0; j < t.outputs.size(); j++) {
                    d[t.outputs[j]]--;
                    if (d[t.outputs[j]] == 0) {
                        q.push(t.outputs[j]);
                    }
                }
            }
        }

        best_lantency = INF_LANTENCY;
    }

    void vector_resize() {
        lantencies.resize(n_layers);
        retain.resize(n_layers);
        tensors_to_retain.resize(n_layers);
        granularities.resize(n_layers);
        traversal_orders.resize(n_layers);
    }

    void get_necessity() {
        n_layers = subgraphs.size();
        d_in.resize(n_layers);
        d_out.resize(n_layers);
        for (int i = 0; i < subgraphs.size(); i++) {
            d_in[i].assign(n, 0);
            d_out[i].assign(n, 0);
            for (int j = 0; j < subgraphs[i].size(); j++) {
                const Node& nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.outputs.size(); k++) {
                    d_in[i][nd.outputs[0]]++;
                }
                for (int k = 0; k < nd.inputs.size(); k++) {
                    d_out[i][nd.inputs[k]]++;
                }
            }
        }
        vector_resize();
    }

    int fd_fa(int x, std::vector<int>& fa) {
        if (x != fa[x]) {
            fa[x] = fd_fa(fa[x], fa);
        }
        return fa[x];
    }

    bool get_subgraphs() {
        std::vector<int> fa;
        for (int i = 0; i < m; i++) {
            fa.push_back(i);
        }
        for (int i = 0; i < n; i++) {
            const Tensor& t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (!is_cut[i]) {
                int x = fd_fa(t.input, fa);
                for (int j = 0; j < t.outputs.size(); j++) {
                    fa[fd_fa(t.outputs[j], fa)] = x;
                }
            }
        }
        std::vector<int> d(m, 0);
        std::vector<std::vector<int> > out;
        out.resize(m);
        for (int i = 0; i < n; i++) {
            const Tensor& t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (is_cut[i]) {
                int x = fd_fa(t.input, fa);
                for (int j = 0; j < t.outputs.size(); j++) {
                    int y = fd_fa(t.outputs[j], fa);
                    if (x != y) {
                        out[x].push_back(y);
                        d[y]++;
                    }
                }
            }
        }
        std::vector<int> order;
        std::deque<int> dq;
        for (int i = 0; i < m; i++) {
            if (fa[i] == i) {
                if (d[i] == 0) {
                    dq.push_front(i);
                }
            }
        }
        while (dq.size()) {
            int x = dq.front();
            dq.pop_front();
            order.push_back(x);
            bool flag = 0;
            for (int i = 0; i < out[x].size(); i++) {
                d[out[x][i]]--;
                if (d[out[x][i]] == 0) {
                    if (!flag) {
                        dq.push_front(out[x][i]);
                        flag = 1;
                    }
                    else {
                        dq.push_back(out[x][i]);
                    }
                }
            }
        }

        std::vector<int> from_order;
        from_order.resize(m);
        for (int i = 0; i < order.size(); i++) {
            from_order[order[i]] = i;
        }
        subgraphs.clear();
        subgraphs.resize(order.size());
        for (int j = 0; j < topo_seq.size(); j++) {
            subgraphs[from_order[fa[topo_seq[j]]]].push_back(topo_seq[j]);
        }

        get_necessity();

        for (int i = 0; i < subgraphs.size(); i++) {
            // if (d_out[i][nodes[subgraphs[i].back()].outputs[0]] != 0) {
            //     for (int j = 0; j < subgraphs[i].size(); j++) {
            //         printf("%d ", subgraphs[i][j]);
            //     }
            //     printf("\n");
            //     exit(0);
            // }
            for (int j = 0; j < subgraphs[i].size() - 1; j++) {
                if (d_out[i][nodes[subgraphs[i][j]].outputs[0]] == 0) {
                    // printf("Multiple outputs!\n");
                    return 0;
                }
            }
        }

        // Check the legitimacy
        std::vector<bool> used_op(m, false), is_computed(n, false);
        for (int i = 0; i < subgraphs.size(); i++) {
            for (int j = 0; j < subgraphs[i].size(); j++) {
                used_op[subgraphs[i][j]] = true;
            }
        }
        for (int i = 0; i < m; i++) {
            if (!used_op[i]) {
                // printf("!used_op[i]\n");
                return 0; // assert
            }
        }
        for (int i = 0; i < n; i++) {
            is_computed[i] = true;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < this->nodes[i].outputs.size(); j++) {
                is_computed[nodes[i].outputs[j]] = false;
            }
        }

        std::vector<bool> ephemeral_computed;
        for (int i = 0; i < subgraphs.size(); i++) {
            ephemeral_computed.assign(n, false);
            for (int j = 0; j < subgraphs[i].size(); j++) {
                const Node& nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.inputs.size(); k++) {
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
                for (int k = 0; k < nd.outputs.size(); k++) {
                    ephemeral_computed[nd.outputs[k]] = true;
                }
            }
            for (int j = 0; j < subgraphs[i].size(); j++) {
                const Node& nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.outputs.size(); k++) {
                    if (d_out[i][nodes[subgraphs[i][j]].outputs[k]] == 0) {
                        is_computed[nodes[subgraphs[i][j]].outputs[k]] = true;
                    }
                }
            }
        }
        return 1;
    }

    void upd_retain(int layer) {
        tensors_to_retain[layer].clear();
        if (retain[layer]) {
            tensors_to_retain[layer].push_back(nodes[subgraphs[layer].back()].outputs[0]);
        }
    }

    void upd_retains() {
        tensors_to_retain.resize(n_layers);
        for (int layer = 0; layer < n_layers; layer++) {
            upd_retain(layer);
        }
    }

    int get_point(int n, int m, int x, int y) {
        return x * m + y;
    }

    double upd_traversal_order(int layer) {
        int n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularities[layer][0];
        int n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularities[layer][1];

        traversal_orders[layer].clear();

        if (n_h > 1 && n_w > 1) {
            double h_lantency, w_lantency;
            traversal_orders[layer].clear();
            for (int i = 0; i < n_h / 2; i++) {
                for (int j = 0; j < n_w; j++) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i, j));
                }
                for (int j = n_w - 1; j >= 0; j--) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i + 1, j));
                }
            }
            h_lantency = get_lantency(layer);

            traversal_orders[layer].clear();
            for (int j = 0; j < n_w / 2; j++) {
                for (int i = 0; i < n_h; i++) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, i, 2 * j));
                }
                for (int i = n_h - 1; i >= 0; i--) {
                    traversal_orders[layer].push_back(get_point(n_h, n_w, i, 2 * j + 1));
                }
            }
            w_lantency = get_lantency(layer);

            if (h_lantency < w_lantency) {
                traversal_orders[layer].clear();
                for (int i = 0; i < n_h / 2; i++) {
                    for (int j = 0; j < n_w; j++) {
                        traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i, j));
                    }
                    for (int j = n_w - 1; j >= 0; j--) {
                        traversal_orders[layer].push_back(get_point(n_h, n_w, 2 * i + 1, j));
                    }
                }
            }
            return std::min(h_lantency, w_lantency);
        }
        else {
            return get_lantency(layer);
        }
    }

    long long cal_memory(std::vector<int>& granularity, int layer) {
        const int n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularity[0];
        const int n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularity[1];
        const int n_k = nodes[subgraphs[layer].back()].is_matmul ? tensors[nodes[subgraphs[layer].back()].inputs[0]].width / granularity[2] : 1;
        for (int i = 0; i < subgraphs[layer].size(); i++) {
            const Node& nd = nodes[subgraphs[layer][i]];
            for (int j = 0; j < nd.inputs.size(); j++) {
                int id = nd.inputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
            }
            for (int j = 0; j < nd.outputs.size(); j++) {
                int id = nd.outputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
            }
        }
        for (int i = subgraphs[layer].size() - 1; i >= 0; i--) {
            const Node& nd = nodes[subgraphs[layer][i]];
            if (i == subgraphs[layer].size() - 1) {
                if (nd.is_matmul) {
                    h[nd.inputs[0]][0] = h[nd.inputs[0]][1] = granularity[0];
                    w[nd.inputs[0]][0] = w[nd.inputs[0]][1] = granularity[2];
                    h[nd.inputs[1]][0] = h[nd.inputs[1]][1] = granularity[2];
                    w[nd.inputs[1]][0] = w[nd.inputs[1]][1] = granularity[1];
                }
                else {
                    for (int j = 0; j < nd.inputs.size(); j++) {
                        h[nd.inputs[j]][0] = h[nd.inputs[j]][1] = granularity[0];
                        w[nd.inputs[j]][0] = w[nd.inputs[j]][1] = granularity[1];
                    }
                }
            }
            else {
                if (nd.is_matmul) {
                    for (int j = 0; j <= 1; j++) {
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
                }
                else {
                    for (int j = 0; j <= 1; j++) {
                        for (int k = 0; k < nd.inputs.size(); k++) {
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

        int retained_input = n;
        long long retain_memory = 0;
        if (layer > 0) {
            for (int i = 0; i < tensors_to_retain[layer - 1].size(); i++) {
                retained_input = tensors_to_retain[layer - 1][i];
                retain_memory += tensors[tensors_to_retain[layer - 1][i]].height * tensors[tensors_to_retain[layer - 1][i]].width;
            }
        }
        for (int i = 0; i < tensors_to_retain[layer].size(); i++) {
            retain_memory += tensors[tensors_to_retain[layer][i]].height * tensors[tensors_to_retain[layer][i]].width;
        }

        long long global_memory = 0;
        for (int i = 0; i < subgraphs[layer].size(); i++) {
            const Node& nd = nodes[subgraphs[layer][i]];
            for (int j = 0; j < nd.inputs.size(); j++) {
                int id = nd.inputs[j];
                if (d_in[layer][id] == 0 && id != retained_input) {
                    if (h[id][0] > h[id][1] && w[id][0] < w[id][1]) {
                        global_memory += h[id][0] * w[id][0] + h[id][1] * w[id][1];
                    }
                    else {
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

    void upd_granularity(int layer) {
        std::vector<int> granularity, best_granularity;
        std::vector<int> best_traversal_order;
        double lantency, best_lantency = INF_LANTENCY;

        const Node& nd = nodes[subgraphs[layer].back()];
        int H_lim_r = tensors[nd.outputs[0]].height;
        int W_lim_r = tensors[nd.outputs[0]].width;
        int H_lim_l = std::max(native_granularity[0] >> 2, 1);
        int W_lim_l = std::max(native_granularity[1] >> 2, 1);

        int K_lim = 1;
        if (nd.is_matmul) {
            K_lim = tensors[nd.inputs[0]].width;
        }

        for (int H = H_lim_r; H >= H_lim_l; H >>= 1) {
            for (int W = W_lim_r; W >= W_lim_l; W >>= 1) {
                granularity = { H, W, 1 };
                if (cal_memory(granularity, layer) <= fast_memory_capacity) {
                    for (int K = K_lim; K >= 1; K >>= 1) {
                        granularity = { H, W, K };
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
        lantencies[layer] = best_lantency;
        granularities[layer] = best_granularity;
        traversal_orders[layer] = best_traversal_order;
    }

    bool upd_granularities() {
        granularities.resize(n_layers);
        for (int layer = 0; layer < n_layers; layer++) {
            upd_granularity(layer);
        }
        return 1;
    }

    double compute_lantency(const Node& nd, int H, int W, int K) {
        if (nd.is_matmul) {
            return (double)nd.cost * std::max(H / native_granularity[0], 1)
                * std::max(W / native_granularity[1], 1)
                / (tensors[nd.inputs[0]].width / K);
        }
        else {
            return (double)nd.cost * std::max(H / native_granularity[0], 1)
                * std::max(W / native_granularity[1], 1);
        }
    }

    double get_lantency(int layer) {
        const int n_h = tensors[nodes[subgraphs[layer].back()].outputs[0]].height / granularities[layer][0];
        const int n_w = tensors[nodes[subgraphs[layer].back()].outputs[0]].width / granularities[layer][1];
        const int n_k = nodes[subgraphs[layer].back()].is_matmul ? tensors[nodes[subgraphs[layer].back()].inputs[0]].width / granularities[layer][2] : 1;

        double lantency = 0.;
        double cal_lantency = 0.;
        for (int i = 0; i < subgraphs[layer].size(); i++) {
            const Node& nd = nodes[subgraphs[layer][i]];
            for (int j = 0; j < nd.inputs.size(); j++) {
                int id = nd.inputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
                is_h[id][0] = is_h[id][1] = false;
            }
            for (int j = 0; j < nd.outputs.size(); j++) {
                int id = nd.outputs[j];
                h[id][0] = h[id][1] = w[id][0] = w[id][1] = 0;
                is_h[id][0] = is_h[id][1] = false;
            }
        }
        for (int i = subgraphs[layer].size() - 1; i >= 0; i--) {
            const Node& nd = nodes[subgraphs[layer][i]];
            int K = 1;
            if (nd.is_matmul) {
                K = (i == subgraphs[layer].size() - 1) ? granularities[layer][2] : tensors[nd.inputs[0]].width;
            }
            if (i == subgraphs[layer].size() - 1) {
                if (nd.is_matmul) {
                    h[nd.inputs[0]][0] = h[nd.inputs[0]][1] = granularities[layer][0];
                    w[nd.inputs[0]][0] = w[nd.inputs[0]][1] = granularities[layer][2];
                    is_h[nd.inputs[0]][0] = is_h[nd.inputs[0]][1] = true;
                    h[nd.inputs[1]][0] = h[nd.inputs[1]][1] = granularities[layer][2];
                    w[nd.inputs[1]][0] = w[nd.inputs[1]][1] = granularities[layer][1];
                    is_h[nd.inputs[1]][0] = is_h[nd.inputs[1]][1] = false;

                    cal_lantency += (double)compute_lantency(nd, granularities[layer][0], granularities[layer][1], K);
                }
                else {
                    for (int j = 0; j < nd.inputs.size(); j++) {
                        h[nd.inputs[j]][0] = h[nd.inputs[j]][1] = granularities[layer][0];
                        w[nd.inputs[j]][0] = w[nd.inputs[j]][1] = granularities[layer][1];
                    }
                    cal_lantency += (double)compute_lantency(nd, granularities[layer][0], granularities[layer][1], K);
                }
            }
            else {
                if (h[nd.outputs[0]][0] > h[nd.outputs[0]][1] && w[nd.outputs[0]][0] < w[nd.outputs[0]][1]) {
                    cal_lantency += (double)compute_lantency(nd, h[nd.outputs[0]][0], w[nd.outputs[0]][0], K);
                    cal_lantency += (double)compute_lantency(nd, h[nd.outputs[0]][1], w[nd.outputs[0]][1], K);
                }
                else {
                    cal_lantency += (double)compute_lantency(nd, std::max(h[nd.outputs[0]][0], h[nd.outputs[0]][1]),
                        std::max(w[nd.outputs[0]][0], w[nd.outputs[0]][1]), K);
                }
                if (nd.is_matmul) {
                    for (int j = 0; j <= 1; j++) {
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
                }
                else {
                    for (int j = 0; j <= 1; j++) {
                        for (int k = 0; k < nd.inputs.size(); k++) {
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
        double part_memory[2] = { 0., 0. };
        double k_memory = 0., full_memory = 0.;
        double part_k_memory[2] = { 0., 0. };

        int retained_input = n;
        if (layer > 0 && retain[layer - 1]) {
            retained_input = nodes[subgraphs[layer - 1].back()].outputs[0];
        }

        for (int i = 0; i < subgraphs[layer].size(); i++) {
            const Node& nd = nodes[subgraphs[layer][i]];
            for (int j = 0; j < nd.inputs.size(); j++) {
                int id = nd.inputs[j];
                if (d_in[layer][id] == 0 && id != retained_input) {
                    if (h[id][0] > h[id][1] && w[id][0] < w[id][1]) {
                        global_memory += h[id][0] * w[id][0] + h[id][1] * w[id][1];
                    }
                    else {
                        global_memory += std::max(h[id][0], h[id][1]) * std::max(w[id][0], w[id][1]);
                    }
                    if ((h[id][0] == tensors[id].height && w[id][0] == tensors[id].width)
                        || (h[id][1] == tensors[id].height && w[id][1] == tensors[id].width)) {
                        full_memory += h[id][0] * w[id][0];
                    }
                    else {
                        if (h[id][0] == h[id][1] && w[id][0] == w[id][1]) {
                            part_memory[is_h[id][0]] += h[id][0] * w[id][0];
                            if ((is_h[id][0] && w[id][0] == tensors[id].width) || (!is_h[id][0] && h[id][0] == tensors[id].height)) {
                                part_k_memory[is_h[id][0]] += h[id][0] * w[id][0];
                            }
                        }
                        else {
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
        const double inv_bandwidth = 1.0 / slow_memory_bandwidth;
        for (int h = 0; h < n_h; h++) {
            for (int w = 0; w < n_w; w++) {
                bool link_h, link_w;
                if (traversal_orders[layer].size() == 0) {
                    if (n_w == 1) {
                        link_h = 0;
                    }
                    else {
                        link_h = (h != 0);
                    }
                    if (n_h == 1) {
                        link_w = 0;
                    }
                    else {
                        link_w = (w != 0);
                    }
                }
                else {
                    if (h == 0 && w == 0) {
                        link_h = link_w = 0;
                    }
                    else {
                        int pre_idx = traversal_orders[layer][h * n_w + w - 1];
                        int pre_h = pre_idx / n_w;
                        int pre_w = pre_idx % n_w;
                        int now_idx = traversal_orders[layer][h * n_w + w];
                        int now_h = now_idx / n_w;
                        int now_w = now_idx % n_w;
                        link_h = (pre_h == now_h);
                        link_w = (pre_w == now_w);
                    }
                }
                if (n_k <= 2) {
                    for (int k = 0; k < n_k; k++) {
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
                                    }
                                    else {
                                        turn_memory -= part_k_memory[0];
                                        turn_memory -= part_k_memory[1];
                                    }
                                }
                                else {
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
                            int id = nodes[subgraphs[layer].back()].outputs[0];
                            if (total_d_out[id] == 0) {
                                if (output_is_retained) {
                                    turn_memory += tensors[id].height * tensors[id].width;
                                }
                            }
                        }
                        // printf("%lf %lf\n", turn_memory / slow_memory_bandwidth, cal_lantency);
                        lantency += std::max(turn_memory * inv_bandwidth, cal_lantency);
                    }
                }
                else {
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
                        int id = nodes[subgraphs[layer].back()].outputs[0];
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

    void upd_lantencies() {
        lantencies.resize(n_layers);
        for (int layer = 0; layer < n_layers; layer++) {
            lantencies[layer] = get_lantency(layer);
        }
    }

    double get_total_lantency() {
        double total_lantency = 0.;
        for (int layer = 0; layer < n_layers; layer++) {
            total_lantency += lantencies[layer];
        }
        return total_lantency;
    }

    std::tuple<double, std::vector<bool> > get_retain() {
        // bottle neck
        // 针对 INF 情况优化
        std::vector<double> lantency = { INF_LANTENCY, INF_LANTENCY };
        std::vector<std::vector<int> > pre;
        std::vector<double> temp_lantency = { INF_LANTENCY, INF_LANTENCY };
        std::vector<int> temp_pre = { 0, 0 };
        for (int layer = 0; layer < n_layers; layer++) {
            if (layer == 0) {
                retain[layer] = 0;
                upd_retain(layer);
                upd_granularity(layer);
                temp_lantency[0] = lantencies[layer];
                temp_pre[0] = 0;

                retain[layer] = 1;
                upd_retain(layer);
                upd_granularity(layer);
                temp_lantency[1] = lantencies[layer];
                temp_pre[1] = 0;
            }
            else {
                if (lantency[0] < INF_LANTENCY / 2) {
                    retain[layer - 1] = 0;
                    upd_retain(layer - 1);

                    retain[layer] = 0;
                    upd_retain(layer);
                    upd_granularity(layer);
                    temp_lantency[0] = lantencies[layer] + lantency[0];
                    temp_pre[0] = 0;

                    retain[layer] = 1;
                    upd_retain(layer);
                    upd_granularity(layer);
                    temp_lantency[1] = lantencies[layer] + lantency[0];
                    temp_pre[1] = 0;
                }
                else {
                    temp_lantency[0] = INF_LANTENCY;
                    temp_pre[0] = 0;
                    temp_lantency[1] = INF_LANTENCY;
                    temp_pre[1] = 0;
                }

                if (lantency[1] < INF_LANTENCY / 2) {
                    retain[layer - 1] = 1;
                    upd_retain(layer - 1);

                    retain[layer] = 0;
                    upd_retain(layer);
                    upd_granularity(layer);

                    if (lantencies[layer] + lantency[1] < temp_lantency[0]) {
                        temp_lantency[0] = lantencies[layer] + lantency[1];
                        temp_pre[0] = 1;
                    }

                    retain[layer] = 1;
                    upd_retain(layer);
                    upd_granularity(layer);
                    if (lantencies[layer] + lantency[1] < temp_lantency[1]) {
                        temp_lantency[1] = lantencies[layer] + lantency[1];
                        temp_pre[1] = 1;
                    }
                }
            }
            lantency = temp_lantency;
            pre.push_back(temp_pre);
        }
        std::vector<bool> retain_strategy;
        bool retain_now = false;
        for (int layer = n_layers - 1; layer >= 0; layer--) {
            retain_strategy.push_back(retain_now);
            retain_now = pre[layer][retain_now];
        }
        std::reverse(retain_strategy.begin(), retain_strategy.end());
        return { lantency[0], retain_strategy };
    }

    void ini(std::vector<std::vector<int> >& subgraphs,
        std::vector<std::vector<int> >& granularities,
        std::vector<std::vector<int> >& tensors_to_retain,
        std::vector<std::vector<int> >& traversal_orders) {
        this->subgraphs = subgraphs;
        get_necessity();
        this->granularities = granularities;
        this->tensors_to_retain = tensors_to_retain;
        retain.resize(n_layers);
        for (int layer = 0; layer < n_layers; layer++) {
            if (tensors_to_retain[layer].empty()) {
                retain[layer] = false;
            }
            else {
                retain[layer] = true;
            }
        }
        this->traversal_orders = traversal_orders;
    }

    std::tuple<std::vector<std::vector<int> >,
        std::vector<std::vector<int> >,
        std::vector<std::vector<int> >,
        std::vector<std::vector<int> >,
        std::vector<double> > final_solution(std::vector<bool>& best_cut) {
        std::vector<bool> temp_cut = is_cut;
        is_cut = best_cut;
        get_subgraphs();

        auto [best_lantency, best_retain] = get_retain();
        retain = best_retain;
        upd_retains();
        upd_granularities();
        upd_lantencies();

        is_cut = temp_cut;
        return { subgraphs, granularities, tensors_to_retain, traversal_orders, lantencies };
    }

    void load_answer() {
        auto [subgraphs, granularities, tensors_to_retain, traversal_orders, lantencies] =
            final_solution(best_cut);

        nlohmann::ordered_json j_out;
        j_out["subgraphs"] = subgraphs;
        j_out["granularities"] = granularities;
        j_out["tensors_to_retain"] = tensors_to_retain;
        for (int i = 0; i < traversal_orders.size(); i++) {
            j_out["traversal_orders"].push_back(
                traversal_orders[i].empty() ? nlohmann::ordered_json(nullptr) : nlohmann::ordered_json(traversal_orders[i])
            );
        }
        j_out["subgraph_latencies"] = lantencies;
        std::ofstream file_out(OUTPUT_PATH);
        file_out << j_out.dump(4);
        file_out.close();
    }

    void upd_answer(double current_lantency, std::vector<bool> current_cut) {
        if (current_lantency < best_lantency) {
            best_lantency = current_lantency;
            best_cut = current_cut;
            load_answer();
        }
    }

    void sa_solution() {
        double best_lantency, current_lantency;
        std::vector<bool> best_cut;
        is_cut.assign(n, true);

        get_subgraphs();

        auto [new_lantency, new_retain] = get_retain();
        best_lantency = current_lantency = new_lantency;
        best_cut = is_cut;

        double init_temp_0 = 30000000.0;
        double min_temp_0 = 0.1;
        double cooling_rate_0 = 0.99;
        int T = 0;
        double num = 0, sum = 0;

        for (double temp_0 = init_temp_0; temp_0 >= min_temp_0;) {
            int x = ran(0, n - 1);
            is_cut[x] = !is_cut[x];

            if (!get_subgraphs()) { is_cut[x] = !is_cut[x]; }
            else {
                temp_0 *= cooling_rate_0;
                double new_lantency;
                if (memo.count(is_cut)) {
                    new_lantency = memo[is_cut];
                }
                else {
                    auto [temp, new_retain] = get_retain();
                    // auto 定义的东西，在退出分支的时候会自动销毁
                    new_lantency = temp;
                    memo[is_cut] = new_lantency;
                }

                if (current_lantency > new_lantency || ran_real(0, 1) < std::exp((current_lantency - new_lantency) / temp_0)) {
                    if (best_lantency > new_lantency) {
                        best_lantency = new_lantency;
                        best_cut = is_cut;
                    }
                    current_lantency = new_lantency;
                }
                else {
                    is_cut[x] = !is_cut[x];
                }
            }

            T++;
            if (T % 500 == 0) {
                upd_answer(best_lantency, best_cut);
            }
        }
        upd_answer(best_lantency, best_cut);
        printf("%lf\n", best_lantency);
    }

    std::vector<double> evaluate(std::vector<std::vector<int> >& subgraphs,
        std::vector<std::vector<int> >& granularities,
        std::vector<std::vector<int> >& tensors_to_retain,
        std::vector<std::vector<int> >& traversal_orders) {
        ini(subgraphs,
            granularities,
            tensors_to_retain,
            traversal_orders);
        upd_lantencies();
        return lantencies;
    }

    bool check_OOM(std::vector<std::vector<int> >& subgraphs,
        std::vector<std::vector<int> >& granularities,
        std::vector<std::vector<int> >& tensors_to_retain,
        std::vector<std::vector<int> >& traversal_orders) {
        ini(subgraphs,
            granularities,
            tensors_to_retain,
            traversal_orders);
        for (int layer = 0; layer < n_layers; layer++) {
            if (cal_memory(granularities[layer], layer) > fast_memory_capacity) {
                return 0;
            }
        }
        return 1;
    }
};

void solve(const std::string INPUT_PATH, const std::string OUTPUT_PATH) {
    std::ifstream file_in(INPUT_PATH);
    json j_in;
    file_in >> j_in;

    std::vector<int> widths = j_in["widths"];
    std::vector<int> heights = j_in["heights"];
    std::vector<std::vector<int> > inputs = j_in["inputs"];
    std::vector<std::vector<int> > outputs = j_in["outputs"];
    std::vector<double> base_costs = j_in["base_costs"];
    std::vector<std::string> op_types = j_in["op_types"];
    long long fast_memory_capacity = j_in["fast_memory_capacity"];
    double slow_memory_bandwidth = j_in["slow_memory_bandwidth"];
    std::vector<int> native_granularity = j_in["native_granularity"];

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

    for (int t = 0; t < 1; t++) {
        myEvaluater.sa_solution();
    }

    // print(subgraphs, granularities, tensors_to_retain, traversal_orders);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "time:" << duration.count() << " ms" << std::endl;
}

void check_out(const std::string INPUT_PATH, const std::string OUTPUT_PATH, bool mode = 0) {
    std::ifstream file_in(INPUT_PATH);
    if (!file_in.is_open()) {
        std::cerr << "Can't open input.json!" << std::endl;
        return;
    }
    json j_in;
    file_in >> j_in;

    std::vector<int> widths = j_in["widths"];
    std::vector<int> heights = j_in["heights"];
    std::vector<std::vector<int> > inputs = j_in["inputs"];
    std::vector<std::vector<int> > outputs = j_in["outputs"];
    std::vector<double> base_costs = j_in["base_costs"];
    std::vector<std::string> op_types = j_in["op_types"];
    long long fast_memory_capacity = j_in["fast_memory_capacity"];
    double slow_memory_bandwidth = j_in["slow_memory_bandwidth"];
    std::vector<int> native_granularity = j_in["native_granularity"];

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

    std::ifstream file_out(OUTPUT_PATH);
    if (!file_out.is_open()) {
        std::cerr << "Can't open output.json!" << std::endl;
        return;
    }

    json j_out;
    file_out >> j_out;

    std::vector<std::vector<int> > subgraphs = j_out["subgraphs"];
    std::vector<std::vector<int> > granularities = j_out["granularities"];
    std::vector<std::vector<int> > tensors_to_retain = j_out["tensors_to_retain"];
    std::vector<std::vector<int> > traversal_orders;
    for (auto& item : j_out["traversal_orders"]) {
        if (item.is_null()) {
            traversal_orders.push_back({});
        }
        else {
            traversal_orders.push_back(item.get<std::vector<int> >());
        }
    }

    if (!myEvaluater.check_OOM(subgraphs, granularities, tensors_to_retain, traversal_orders)) {
        printf("OOM!\n");
        return;
    }

    std::vector<double> lantencies = myEvaluater.evaluate(subgraphs, granularities, tensors_to_retain, traversal_orders);

    if (!mode) {
        std::vector<double> subgraph_latencies = j_out["subgraph_latencies"];
        for (int i = 0; i < lantencies.size(); i++) {
            if (subgraph_latencies[i] != lantencies[i]) {
                printf("Lantency of subgraph %d is wrong!\n", i);
            }
        }
        printf("Lantencies is right!\n");
    }
    else {
        for (int i = 0; i < lantencies.size(); i++) {
            printf("%lf\n", lantencies[i]);
        }
        printf("%lf\n", myEvaluater.get_total_lantency());
    }
}

int main(int argc, char* argv[]) {
    std::cout << "case: " << argv[1] << std::endl;

    INPUT_PATH = argv[1];
    OUTPUT_PATH = argv[2];

    solve(argv[1], argv[2]);

    // INPUT_PATH = "input.json";
    // OUTPUT_PATH = "output.json";
    // solve("input.json", "output.json");

    // check_out("input.json", "output.json", 1);
    return 0;
}