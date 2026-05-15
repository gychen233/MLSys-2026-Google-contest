#pragma GCC optimize("Ofast")
#include<bits/stdc++.h>

#define INF 2000000000
#define MAX_M 300
#define MAX_N 600
#define MAX_SIZ 32
#define INF_LANTENCY 2000000000.0
#define init(a); memset(a, 0, sizeof(a));

std::mt19937 rnd(std::chrono::steady_clock::now().time_since_epoch().count());
#define ran(l, r) std::uniform_int_distribution<int>(l, r)(rnd)
#define ran_real(l, r) std::uniform_real_distribution<float>(l, r)(rnd)

int div_ceil(int n, int m) {
    return (n + m - 1) / m;
}

struct Tensor {
    int height;
    int width;
    Tensor(): height(0), width(0) {}
    Tensor(int height, int width) {
        this->height = height;
        this->width = width;
    }
};

struct TensorNode {
    int height;
    int width;
    int input;
    std::vector<int> outputs;
    bool is_cut, is_retained;
    bool is_input;
};

struct Node {
    float cost;
    bool is_matmul;
    std::vector<int> inputs;
    std::vector<int> outputs;
};

struct Range {
    int l, r;
    Range(): l(0), r(0) {}
    Range(int l, int r) {
        this->l = l;
        this->r = r;
    }
};

struct LoadingData {
    int id;
    Range height, width;
    int siz;
    bool is_load;
    LoadingData(): id(0), height(Range(0, 0)), width(Range(0, 0)), siz(0) {}
    LoadingData(int id, Range height, Range width, int siz) {
        this->id = id;
        this->height = height;
        this->width = width;
        this->siz = siz;
    }
    LoadingData(int id, Range height, Range width) {
        this->id = id;
        this->height = height;
        this->width = width;
        this->siz = (height.r - height.l) * (width.r - width.l);
    }
    LoadingData(int id, int height_l, int height_r, int width_l, int width_r) {
        this->id = id;
        this->height = Range(height_l, height_r);
        this->width = Range(width_l, width_r);
        this->siz = (height_r - height_l) * (width_r - width_l);
    }
};

int gcd(int x,int y) {
    if (y == 0) return x;
    else return gcd(y, x % y);
}

Tensor merge(Tensor x, LoadingData y) {
    return Tensor(gcd(x.height, y.height.r - y.height.l), gcd(x.width, y.width.r - y.width.l));
}

struct DataPosition {
    int id, h, w;
    int siz;
    DataPosition(): id(0), h(0), w(0), siz(0) {}
    DataPosition(int id, int h, int w, int siz) {
        this->id = id;
        this->h = h;
        this->w = w;
        this->siz = siz;
    }
    bool operator<(const DataPosition& other) const {
        return id == other.id ? (h == other.h ? w < other.w : h < other.h) : id < other.id;
    }
};

class Belady {
private:
    std::vector<std::vector<int> > tim[MAX_SIZ][MAX_SIZ];
    std::vector<int> pos[MAX_SIZ][MAX_SIZ];

    int next_tim(DataPosition x) {
        return tim[x.h][x.w][x.id].size() > pos[x.h][x.w][x.id]
             ? tim[x.h][x.w][x.id][pos[x.h][x.w][x.id]]
             : INF;
    }

    std::priority_queue<std::pair<int, DataPosition> > q;
    std::set<DataPosition> st;
    int fast_memory_capacity;
    int used_memory_capacity;

public:
    Belady (int n, std::vector<LoadingData>& memory_trace) {
        for (int i = 0; i < MAX_SIZ; i++) {
            for (int j = 0; j < MAX_SIZ; j++) {
                tim[i][j].resize(n);
                pos[i][j].assign(n, 0); // 注意初始化为全零
            }
        }
        for (int i = 0; i < memory_trace.size(); i++) {
            LoadingData dat = memory_trace[i];
            for (int h = dat.height.l; h < dat.height.r; h++) {
                for (int w = dat.width.l; w < dat.width.r; w++) {
                    tim[h][w][dat.id].push_back(i);
                }
            }
        }
    }

    void reset(int fast_memory_capacity) {
        this->fast_memory_capacity = fast_memory_capacity;
        used_memory_capacity = 0;
        st.clear();
        while(q.size()) {
            q.pop();
        }
    }

    void pop(DataPosition x) {
        pos[x.h][x.w][x.id]++;
    }

    float ins(DataPosition x) {
        if (st.count(x)) {
            pos[x.h][x.w][x.id]++;
            q.push(std::make_pair(next_tim(x), x));
            return 0;
        }
        // printf("%d %d %d\n", x.siz, fast_memory_capacity, q.size());
        while (used_memory_capacity + x.siz > fast_memory_capacity) {
            int dat_tim = q.top().first;
            DataPosition dat = q.top().second;
            q.pop();
            if (dat_tim == next_tim(dat)) {
                st.erase(dat);
                used_memory_capacity -= dat.siz;
            }
        }
        used_memory_capacity += x.siz;
        pos[x.h][x.w][x.id]++;
        st.insert(x);
        q.push(std::make_pair(next_tim(x), x));
        return (float)x.siz;
    }
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
    // for (int i = 0; i < traversal_orders.size(); i++) {
    //     for (int j = 0; j < traversal_orders[i].size(); j++) {
    //         printf("%d ", traversal_orders[i][j]);
    //     }
    //     printf("\n");
    // }
}

class ScheduleEvaluator {
private:
    int n, m;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
    int fast_memory_capacity;
    float slow_memory_bandwidth;
    std::vector<int> native_granularity;
public:
    ScheduleEvaluator() : n(0), m(0), fast_memory_capacity(0), slow_memory_bandwidth(0.) {}

    ScheduleEvaluator (
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        const std::vector<std::vector<int> >& inputs,
        const std::vector<std::vector<int> >& outputs,
        const std::vector<float>& base_costs,
        const std::vector<std::string>& op_types,
        const int fast_memory_capacity,
        const float slow_memory_bandwidth,
        const std::vector<int>& native_granularity
    ) {
        this->fast_memory_capacity = fast_memory_capacity;
        this->slow_memory_bandwidth = slow_memory_bandwidth;
        this->native_granularity = native_granularity;

        this->n = widths.size();
        assert(this->n == heights.size());
        for (int i = 0; i < this->n; i++) {
            Tensor t;
            t.width = widths[i];
            t.height = heights[i];
            this->tensors.push_back(t);
        }

        this->m = inputs.size();
        assert(this->m == outputs.size());
        assert(this->m == base_costs.size());
        assert(this->m == op_types.size());
        for (int i = 0; i < this->m; i++) {
            Node o;
            o.inputs = inputs[i];
            o.outputs = outputs[i];
            o.cost = base_costs[i];
            if (op_types[i] == "MatMul") o.is_matmul = 1;
            else o.is_matmul = 0;
            this->nodes.push_back(o);
        }
    }

    std::vector<float> evaluate(
        std::vector<std::vector<int> >& input_subgraphs,
        std::vector<std::vector<int> >& granularities,
        std::vector<std::vector<int> >& tensors_to_retain,
        std::vector<std::vector<int> >& traversal_orders
    ) {
        std::vector<float> lantency;

        // Get necesssary data
        std::vector<std::vector<int> > subgraphs = input_subgraphs;

        int n_layers = subgraphs.size();
        if (n_layers != granularities.size()) {
            printf("n_layers != granularities.size()\n");
            return lantency; // assert
        }
        if (n_layers != tensors_to_retain.size()) {
            printf("n_layers != tensors_to_retain.size()\n");
            return lantency; // assert
        }
        if (n_layers != traversal_orders.size()) {
            printf("n_layers != traversal_orders.size()\n");
            return lantency; // assert
        }

        int d_in[MAX_N][MAX_N];
        int d_out[MAX_N][MAX_N];
        int total_d_in[MAX_N];
        int total_d_out[MAX_N]; // only used to check whether it is a global ouput
        for (int i = 0; i < subgraphs.size(); i++) {
            init(d_in[i]);
            init(d_out[i]);
            for (int j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.outputs.size(); k++) {
                    d_in[i][nd.outputs[k]]++;
                    total_d_in[nd.outputs[k]]++;
                }
                for (int k = 0; k < nd.inputs.size(); k++) {
                    d_out[i][nd.inputs[k]]++;
                    total_d_out[nd.inputs[k]]++;
                }
            }
        }

        // Check the legitimacy
        bool used_op[MAX_M];
        bool is_computed[MAX_N];
        for (int i = 0; i < subgraphs.size(); i++) {
            for (int j = 0; j < subgraphs[i].size(); j++) {
                used_op[subgraphs[i][j]] = 1;
            }
        }
        for (int i = 0; i < m; i++) {
            if (!used_op[i]) {
                printf("!used_op[i]\n");
                return lantency; // assert
            }
        }
        for (int i = 0; i < n; i++) {
            is_computed[i] = 1;
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < this->nodes[i].outputs.size(); j++) {
                is_computed[nodes[i].outputs[j]] = 0;
            }
        }

        bool ephemeral_computed[MAX_N];
        for (int i = 0; i < subgraphs.size(); i++) {
            init(ephemeral_computed);
            for (int j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.inputs.size(); k++) {
                    if (d_in[i][nd.inputs[k]] > 0) {
                        if (!ephemeral_computed[nd.inputs[k]]) {
                            // printf("!ephemeral_computed[nd.inputs[k]]\n");
                            return lantency; // assert
                        }
                    }
                    else {
                        if (!is_computed[nd.inputs[k]]) {
                            // printf("!is_computed[nd.inputs[k]]\n");
                            return lantency; // assert
                        }
                    }
                }
                for (int k = 0; k < nd.outputs.size(); k++) {
                    ephemeral_computed[nd.outputs[k]] = 1;
                }
            }
            for (int j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.outputs.size(); k++) {
                    if (d_out[i][nodes[subgraphs[i][j]].outputs[k]] == 0) {
                        is_computed[nodes[subgraphs[i][j]].outputs[k]] = 1;
                    }
                }
            }
        }

        // Improve the inner order of subgraphs
        std::vector<int> temp_subgraphs;
        int boundary[MAX_M];
        for (int i = 0; i < subgraphs.size(); i++) {
            temp_subgraphs.clear();
            for (int j = 0; j < subgraphs[i].size(); j++) {
                if (d_out[i][nodes[subgraphs[i][j]].outputs[0]] != 0) {
                    temp_subgraphs.push_back(subgraphs[i][j]);
                }
            }
            boundary[i] = temp_subgraphs.size();
            for (int j = 0; j < subgraphs[i].size(); j++) {
                if (d_out[i][nodes[subgraphs[i][j]].outputs[0]] == 0) {
                    temp_subgraphs.push_back(subgraphs[i][j]);
                }
            }
            subgraphs[i] = temp_subgraphs;
        }

        // OOM check
        std::vector<std::vector<Tensor> > single_compute_size;
        single_compute_size.resize(n);
        int retain_memory, output_memory, temp_memory, conflict_memory;
        bool retained_tensor[MAX_N];
        for (int layer = 0; layer < n_layers; layer++) {
            retain_memory = 0;
            init(retained_tensor);
            for (int i = std::max(layer - 1, 0); i <= layer; i++) {
                for (int j = 0; j < tensors_to_retain[i].size(); j++) {
                    retained_tensor[tensors_to_retain[i][j]] = 1;
                    retain_memory += tensors[tensors_to_retain[i][j]].height * tensors[tensors_to_retain[i][j]].width;
                }
            }
            int H = granularities[layer][0];
            int W = granularities[layer][1];
            int K = granularities[layer][2];
            for (int i = 0; i < n; i++) {
                single_compute_size[i].clear();
            }

            output_memory = 0;
            conflict_memory = 0;
            for (int i = boundary[layer]; i < subgraphs[layer].size(); i++) {
                Node nd = nodes[subgraphs[layer][i]];
                if (nd.is_matmul && tensors[nd.inputs[0]].width > K) {
                    output_memory += H * W;
                } else {
                    if (nd.is_matmul) {
                        conflict_memory = std::max(conflict_memory, granularities[layer][0] * granularities[layer][1]
                                        + (granularities[layer][0] + granularities[layer][1]) * granularities[layer][2]);
                    } else {
                        conflict_memory = std::max(conflict_memory, granularities[layer][0] * granularities[layer][1] * 2);
                    }
                }
            }

            for (int i = subgraphs[layer].size() - 1; i >= 0; i--) {
                Node nd = nodes[subgraphs[layer][i]];
                for (int j = 0; j < nd.outputs.size(); j++) {
                    int output_id = nd.outputs[j];
                    single_compute_size[output_id].push_back(Tensor(H, W));
                    while(single_compute_size[output_id].size()) {
                        Tensor dat = single_compute_size[output_id].back();
                        single_compute_size[output_id].pop_back();
                        temp_memory = 0;
                        if (nd.is_matmul) {
                            int cal_h = dat.height;
                            int cal_w = dat.width;
                            int cal_k = (i >= boundary[layer] ? K : tensors[nd.inputs[0]].width);
                            if (d_in[layer][nd.inputs[0]]) {
                                single_compute_size[nd.inputs[0]].push_back(Tensor(cal_h, cal_k));
                            }
                            else {
                                if (!retained_tensor[nd.inputs[0]]) {
                                    temp_memory += cal_h * cal_k;
                                }
                            }
                            if (d_in[layer][nd.inputs[1]]) {
                                single_compute_size[nd.inputs[1]].push_back(Tensor(cal_k, cal_w));
                            }
                            else {
                                if (!retained_tensor[nd.inputs[1]]) {
                                    temp_memory += cal_k * cal_w;
                                }
                            }
                        }
                        else {
                            for (int l = 0; l < nd.inputs.size(); l++) {
                                if (d_in[layer][nd.inputs[l]]) {
                                    single_compute_size[nd.inputs[l]].push_back(Tensor(dat.height, dat.width));
                                }
                                else {
                                    if (!retained_tensor[nd.inputs[l]]) {
                                        temp_memory += dat.height * dat.width;
                                    }
                                }
                            }
                        }
                        if (std::max(retain_memory + output_memory + temp_memory, conflict_memory) > fast_memory_capacity) {
                            // printf("Out of Memory!\n");
                            return lantency; // assert
                        }
                    }
                }
            }
        }

        // Trace generation
        // Native_granularity check
        // Calculate the compute time
        // int d_in[MAX_N];
        // int d_out[MAX_N];
        std::vector<LoadingData> memory_trace;
        std::vector<LoadingData> requests[MAX_N];
        int layer_boundary[MAX_N];
        float turn_compute_lantency;
        std::vector<float> compute_lantency;
        std::vector<int> turn_boundary;
        std::vector<bool> memory_out;
        for (int layer = 0; layer < n_layers; layer++) {
            init(retained_tensor);
            for (int i = std::max(layer - 1, 0); i <= layer; i++) {
                for (int j = 0; j < tensors_to_retain[i].size(); j++) {
                    retained_tensor[tensors_to_retain[i][j]] = 1;
                }
            }

            int H = granularities[layer][0];
            int W = granularities[layer][1];
            int K = granularities[layer][2];
            int n_h = 0, n_w = 0, n_k = 1;
            n_h = div_ceil(tensors[nodes[subgraphs[layer].back()].outputs[0]].height, H);
            n_w = div_ceil(tensors[nodes[subgraphs[layer].back()].outputs[0]].width, W);
            for (int i = boundary[layer]; i < subgraphs[layer].size(); i++) {
                Node nd = nodes[subgraphs[layer][i]];
                if (nd.is_matmul) {
                    n_k = std::max(n_k, div_ceil(tensors[nd.inputs[0]].width, K));
                }
            }

            std::vector<LoadingData> layer_trace;
            for (int h = 0; h < n_h; h++) {
                for (int w = 0; w < n_w; w++) {
                    for (int k = 0; k < n_k; k++) {
                        // 这一段 clear 理应可以删去
                        layer_trace.clear();
                        for (int i = 0; i < subgraphs[layer].size(); i++) {
                            Node nd = nodes[subgraphs[layer][i]];
                            for (int j = 0; j < nd.inputs.size(); j++) {
                                requests[nd.inputs[j]].clear();
                            }
                            for (int j = 0; j < nd.outputs.size(); j++) {
                                requests[nd.outputs[j]].clear();
                            }
                        }
                        turn_compute_lantency = 0.;

                        for (int i = subgraphs[layer].size() - 1; i >= 0; i--) {
                            Node nd = nodes[subgraphs[layer][i]];
                            for (int j = 0; j < nd.outputs.size(); j++) {
                                int output_id = nd.outputs[j];
                                if (i >= boundary[layer]) {
                                    if (traversal_orders[layer].size() > 0) {
                                        int pos = h * n_w + w;
                                        int hh = traversal_orders[layer][pos] / n_w;
                                        int ww = traversal_orders[layer][pos] % n_w;
                                        requests[output_id].push_back(LoadingData(output_id, hh * H, hh * H + H, ww * W, ww * W + W));
                                        if (!retained_tensor[output_id]) {
                                            if (!nd.is_matmul || tensors[nd.inputs[0]].width == granularities[layer][2]) {
                                                layer_trace.push_back(LoadingData(output_id, hh * H, hh * H + H, ww * W, ww * W + W));
                                            }
                                        }
                                    } else {
                                        requests[output_id].push_back(LoadingData(output_id, h * H, h * H + H, w * W, w * W + W));
                                        if (!retained_tensor[output_id]) {
                                            if (!nd.is_matmul || tensors[nd.inputs[0]].width == granularities[layer][2]) {
                                                layer_trace.push_back(LoadingData(output_id, h * H, h * H + H, w * W, w * W + W));
                                            }
                                        }
                                    }
                                    
                                }

                                while(requests[output_id].size()) {
                                    LoadingData dat = requests[output_id].back();
                                    requests[output_id].pop_back();
                                    if (nd.is_matmul) {
                                        turn_compute_lantency += nd.cost / n_k;
                                        if (i < boundary[layer] || k < tensors[nd.inputs[0]].width / K) {
                                            Range cal_h = dat.height;
                                            Range cal_w = dat.width;
                                            Range cal_k = (i >= boundary[layer] ? Range(k * K, k * K + K) : Range(0, tensors[nd.inputs[0]].width));
                                            if (cal_h.r - cal_h.l > native_granularity[0] && cal_w.r - cal_w.l <= native_granularity[1]) {
                                                printf("Out of Native Granularity!\n");
                                                return lantency; // assert
                                            }
                                            if (d_in[layer][nd.inputs[0]]) {
                                                requests[nd.inputs[0]].push_back(LoadingData(nd.inputs[0], cal_h, cal_k));
                                            }
                                            else {
                                                if (!retained_tensor[nd.inputs[0]]) {
                                                    layer_trace.push_back(LoadingData(nd.inputs[0], cal_h, cal_k));
                                                }
                                            }
                                            if (d_in[layer][nd.inputs[1]]) {
                                                requests[nd.inputs[1]].push_back(LoadingData(nd.inputs[1], cal_k, cal_w));
                                            }
                                            else {
                                                if (!retained_tensor[nd.inputs[0]]) {
                                                    layer_trace.push_back(LoadingData(nd.inputs[1], cal_k, cal_w));
                                                }
                                            }
                                        }
                                    }
                                    else {
                                        turn_compute_lantency += nd.cost;
                                        if (dat.height.r - dat.height.l > native_granularity[0] && dat.width.r - dat.width.l <= native_granularity[1]) {
                                            printf("Out of Native Granularity!\n");
                                            return lantency; // assert
                                        }
                                        for (int l = 0; l < nd.inputs.size(); l++) {
                                            if (d_in[layer][nd.inputs[l]] > 0) {
                                                requests[nd.inputs[l]].push_back(LoadingData(nd.inputs[l], dat.height, dat.width));
                                            }
                                            else {
                                                if (!retained_tensor[nd.inputs[l]]) {
                                                    layer_trace.push_back(LoadingData(nd.inputs[l], dat.height, dat.width));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        while(layer_trace.size()) {
                            memory_trace.push_back(layer_trace.back());
                            layer_trace.pop_back();
                        }

                        turn_boundary.push_back(memory_trace.size());
                        compute_lantency.push_back(turn_compute_lantency);
                        if (k == n_k - 1) {
                            memory_out.push_back(1);
                        } else {
                            memory_out.push_back(0);
                        }
                    }
                }
            }
            layer_boundary[layer] = turn_boundary.size();
        }

        Tensor siz[MAX_N];
        for (int i = 0; i < memory_trace.size(); i++) {
            siz[memory_trace[i].id] = merge(siz[memory_trace[i].id], memory_trace[i]);
        }

        for (int i = 0; i < memory_trace.size(); i++) {
            memory_trace[i].height.l /= siz[memory_trace[i].id].height;
            memory_trace[i].height.r /= siz[memory_trace[i].id].height;
            memory_trace[i].width.l /= siz[memory_trace[i].id].width;
            memory_trace[i].width.r /= siz[memory_trace[i].id].width;
        }

        // Belady
        Belady myBelady(n, memory_trace);
        int used_memory_capacity, retained_memory_capacity, out_memory_capacity;
        float turn_memory_lantency;
        float layer_lantency;
        for (int layer = 0; layer < subgraphs.size(); layer++) {
            // printf("layer %d:\n", layer);
            int layer_l = (layer == 0 ? 0 : layer_boundary[layer - 1]);
            int layer_r = layer_boundary[layer];
            used_memory_capacity = 0;
            init(retained_tensor);
            if (layer > 0) {
                for (int i = 0; i < tensors_to_retain[layer - 1].size(); i++) {
                    retained_tensor[tensors_to_retain[layer - 1][i]] = 1;
                    used_memory_capacity += tensors[tensors_to_retain[layer - 1][i]].height
                        * tensors[tensors_to_retain[layer - 1][i]].width;
                }
            }
            retained_memory_capacity = 0;
            for (int i = 0; i < tensors_to_retain[layer].size(); i++) {
                retained_tensor[tensors_to_retain[layer][i]] = 1;
                used_memory_capacity += tensors[tensors_to_retain[layer][i]].height
                    * tensors[tensors_to_retain[layer][i]].width;
                if (total_d_in[tensors_to_retain[layer][i]] > 0 && total_d_out[tensors_to_retain[layer][i]] == 0) {
                    retained_memory_capacity += tensors[tensors_to_retain[layer][i]].height
                        * tensors[tensors_to_retain[layer][i]].width;
                }
            }
            out_memory_capacity = 0;
            for (int i = boundary[layer]; i < subgraphs[layer].size(); i++) {
                Node nd = nodes[subgraphs[layer][i]];
                if (nd.is_matmul && tensors[nd.inputs[0]].width > granularities[layer][2]) {
                    for (int j = 0; j < nd.outputs.size(); j++) {
                        if (!retained_tensor[nd.outputs[j]]) {
                            out_memory_capacity += granularities[layer][0] * granularities[layer][1];
                        }
                    }
                }
            }
            used_memory_capacity += out_memory_capacity;
            myBelady.reset(fast_memory_capacity - used_memory_capacity);

            layer_lantency = 0.;
            for (int i = layer_l; i < layer_r; i++) {
                int turn_l = (i == 0 ? 0 : turn_boundary[i - 1]);
                int turn_r = turn_boundary[i];
                turn_memory_lantency = 0.;
                for (int j = turn_l; j < turn_r; j++) {
                    LoadingData dat = memory_trace[j];
                    for (int h = dat.height.l; h < dat.height.r; h++) {
                        for (int w = dat.width.l; w < dat.width.r; w++) {
                            if (retained_tensor[dat.id]) {
                                myBelady.pop(DataPosition(dat.id, h, w, siz[dat.id].height * siz[dat.id].width));
                            }
                            else {
                                turn_memory_lantency += myBelady.ins(DataPosition(dat.id, h, w, siz[dat.id].height * siz[dat.id].width));
                            }
                        }
                    }
                }
                if (memory_out[i]) {
                    turn_memory_lantency += out_memory_capacity;
                }
                if (i == layer_r - 1) {
                    turn_memory_lantency += retained_memory_capacity;
                }
                layer_lantency += std::max(compute_lantency[i], turn_memory_lantency / slow_memory_bandwidth);
            }
            lantency.push_back(layer_lantency);
        }

        return lantency;
    }
};

void check(std::vector<std::vector<int> > subgraphs,
           std::vector<std::vector<int> > granularities,
           std::vector<std::vector<int> > tensors_to_retain,
           std::vector<std::vector<int> > traversal_orders,
           ScheduleEvaluator myEvaluater) {
    print(subgraphs,
            granularities,
            tensors_to_retain,
            traversal_orders);
    std::vector<float> lantency = myEvaluater.evaluate(
                                    subgraphs,
                                    granularities,
                                    tensors_to_retain,
                                    traversal_orders
                                );
    float total_lantencies = 0.;
    for (int i = 0; i < lantency.size(); i++) {
        total_lantencies += lantency[i];
        printf("%lf\n", lantency[i]);
    }
    printf("%lf\n", total_lantencies);
}

class ComputeGraph {
private:
    int n, m;
    std::vector<TensorNode> tensors;
    std::vector<Node> nodes;
    int fast_memory_capacity;
    float slow_memory_bandwidth;
    std::vector<int> native_granularity;
    std::vector<int> topo_id;
    ScheduleEvaluator myEvaluater;
public:
    ComputeGraph (
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        const std::vector<std::vector<int> >& inputs,
        const std::vector<std::vector<int> >& outputs,
        const std::vector<float>& base_costs,
        const std::vector<std::string>& op_types,
        const int fast_memory_capacity,
        const float slow_memory_bandwidth,
        const std::vector<int>& native_granularity,
        ScheduleEvaluator& myEvaluater
    ) {
        this->fast_memory_capacity = fast_memory_capacity;
        this->slow_memory_bandwidth = slow_memory_bandwidth;
        this->native_granularity = native_granularity;
        this->myEvaluater = myEvaluater;

        n = widths.size();
        assert(n == heights.size());
        for (int i = 0; i < n; i++) {
            TensorNode t;
            t.width = widths[i];
            t.height = heights[i];
            t.is_cut = 0;
            t.is_input = 1;
            tensors.push_back(t);
        }

        m = inputs.size();
        assert(m == outputs.size());
        assert(m == base_costs.size());
        assert(m == op_types.size());
        for (int i = 0; i < m; i++) {
            Node o;
            o.inputs = inputs[i];
            o.outputs = outputs[i];
            o.cost = base_costs[i];
            if (op_types[i] == "MatMul") o.is_matmul = 1;
            else o.is_matmul = 0;
            nodes.push_back(o);
            for (int j = 0; j < o.inputs.size(); j++) {
                tensors[o.inputs[j]].outputs.push_back(i);
            }
            for (int j = 0; j < o.outputs.size(); j++) {
                tensors[o.outputs[j]].input = i;
                tensors[o.outputs[j]].is_input = 0;
            }
        }

        topo_id.resize(m);
        int d_in[MAX_N];
        init(d_in);
        for (int i = 0; i < n; i++) {
            TensorNode t = tensors[i];
            for (int j = 0; j < t.outputs.size(); j++) {
                d_in[t.outputs[j]]++;
            }
        }
        std::queue<int> q;
        for (int i = 0; i < m; i++) {
            if (d_in[i] == 0) {
                q.push(i);
            }
        }
        int cnt = 0;
        while(q.size()) {
            int x = q.front();
            q.pop();
            topo_id[x] = ++cnt;
            Node nd = nodes[x];
            for (int i = 0; i < nodes[x].outputs.size(); i++) {
                TensorNode t = tensors[nodes[x].outputs[i]];
                for (int j = 0; j < t.outputs.size(); j++) {
                    d_in[t.outputs[j]]--;
                    if (d_in[t.outputs[j]] == 0) {
                        q.push(t.outputs[j]);
                    }
                }
            }
        }
    }

    int get_point(int n, int m, int x, int y) {
        return x * m + y;
    }

    std::vector<int> get_traversal_order(std::vector<int>& subgraph,
                                         std::vector<int>& granularity) {
        int n_h = 0;
        int n_w = 0;
        int d_out[MAX_N];
        init(d_out);
        for (int i = 0; i < subgraph.size(); i++) {
            Node nd = nodes[subgraph[i]];
            for (int j = 0; j < nd.inputs.size(); j++) {
                d_out[nd.inputs[j]]++;
            }
        }

        for (int i = 0; i < subgraph.size(); i++) {
            if (d_out[nodes[subgraph[i]].outputs[0]] == 0) {
                n_h = std::max(n_h, tensors[nodes[subgraph[i]].outputs[0]].height);
                n_w = std::max(n_h, tensors[nodes[subgraph[i]].outputs[0]].width);
            }
        }

        n_h /= granularity[0];
        n_w /= granularity[1];

        std::vector<int> traversal_order;
        if (n_h == 1 || n_w == 1) return traversal_order;
        for (int i = 0; i < n_h / 2; i++) {
            for (int j = 0; j < n_w / 2; j++) {
                if (i & 1) {
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i, n_w - 2 * j - 1));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i + 1, n_w - 2 * j - 1));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i + 1, n_w - 2 * j - 2));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i, n_w - 2 * j - 2));
                }
                else {
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i, 2 * j));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i + 1, 2 * j));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i + 1, 2 * j + 1));
                    traversal_order.push_back(get_point(n_h, n_w, 2 * i, 2 * j + 1));
                }
            }
        }
        return traversal_order;
    }

    int cal_memory(std::vector<int>& subgraph,
                   std::vector<int>& granularity,
                   std::vector<int>& input_tensors_to_retain,
                   std::vector<int>& output_tensors_to_retain,
                   std::vector<int>& d_in,
                   std::vector<int>& d_out,
                   int boundary) {
        std::vector<std::vector<Tensor> > single_compute_size;
        single_compute_size.resize(n);
        int retain_memory, output_memory, temp_memory, conflict_memory;
        bool retained_tensor[MAX_N];

        retain_memory = 0;
        init(retained_tensor);
        for (int i = 0; i < input_tensors_to_retain.size(); i++) {
            retained_tensor[input_tensors_to_retain[i]] = 1;
            retain_memory += tensors[input_tensors_to_retain[i]].height * tensors[input_tensors_to_retain[i]].width;
        }
        for (int i = 0; i < output_tensors_to_retain.size(); i++) {
            retained_tensor[output_tensors_to_retain[i]] = 1;
            retain_memory += tensors[output_tensors_to_retain[i]].height * tensors[output_tensors_to_retain[i]].width;
        }

        int H = granularity[0];
        int W = granularity[1];
        int K = granularity[2];
        for (int i = 0; i < n; i++) {
            single_compute_size[i].clear();
        }

        output_memory = 0;
        conflict_memory = 0;
        for (int i = boundary; i < subgraph.size(); i++) {
            Node nd = nodes[subgraph[i]];
            if (nd.is_matmul && tensors[nd.inputs[0]].width > K) {
                output_memory += H * W;
            }
            else {
                if (nd.is_matmul) {
                    conflict_memory = std::max(conflict_memory, granularity[0] * granularity[1]
                                    + (granularity[0] + granularity[1]) * granularity[2]);
                } else {
                    conflict_memory = std::max(conflict_memory, granularity[0] * granularity[1] * 2);
                }
            }
        }

        for (int i = subgraph.size() - 1; i >= 0; i--) {
            Node nd = nodes[subgraph[i]];
            for (int j = 0; j < nd.outputs.size(); j++) {
                int output_id = nd.outputs[j];
                single_compute_size[output_id].push_back(Tensor(H, W));
                while(single_compute_size[output_id].size()) {
                    Tensor dat = single_compute_size[output_id].back();
                    single_compute_size[output_id].pop_back();
                    temp_memory = 0;
                    if (nd.is_matmul) {
                        int cal_h = dat.height;
                        int cal_w = dat.width;
                        int cal_k = (i >= boundary ? K : tensors[nd.inputs[0]].width);
                        if (d_in[nd.inputs[0]]) {
                            single_compute_size[nd.inputs[0]].push_back(Tensor(cal_h, cal_k));
                        }
                        else {
                            if (!retained_tensor[nd.inputs[0]]) {
                                temp_memory += cal_h * cal_k;
                            }
                        }
                        if (d_in[nd.inputs[1]]) {
                            single_compute_size[nd.inputs[1]].push_back(Tensor(cal_k, cal_w));
                        }
                        else {
                            if (!retained_tensor[nd.inputs[1]]) {
                                temp_memory += cal_k * cal_w;
                            }
                        }
                    }
                    else {
                        for (int l = 0; l < nd.inputs.size(); l++) {
                            if (d_in[nd.inputs[l]]) {
                                single_compute_size[nd.inputs[l]].push_back(Tensor(dat.height, dat.width));
                            }
                            else {
                                if (!retained_tensor[nd.inputs[l]]) {
                                    temp_memory += dat.height * dat.width;
                                }
                            }
                        }
                    }
                }
            }
        }
        return std::max(retain_memory + output_memory + temp_memory, conflict_memory);
    }

    std::vector<int> get_retain(std::vector<int>& subgraph, int boundary) { 
        std::vector<int> retain;
        for (int i = boundary; i < subgraph.size(); i++) {
            Node nd = nodes[subgraph[i]];
            for (int j = 0; j < nd.outputs.size(); j++) {
                if (tensors[nd.outputs[j]].is_retained) {
                    retain.push_back(nd.outputs[j]);
                }
            }
        }
        std::sort(retain.begin(), retain.end());
        auto it = std::unique(retain.begin(), retain.end());
        retain.erase(it, retain.end());
        return retain;
    }

#define cal_memory(granularity) cal_memory(subgraph, granularity, input_tensors_to_retain, output_tensors_to_retain, d_in, d_out, boundary)
    std::vector<int> get_granularity(std::vector<int>& subgraph,
                                     int boundary,
                                     std::vector<int>& d_in,
                                     std::vector<int>& d_out,
                                     std::vector<int>& input_tensors_to_retain,
                                     std::vector<int>& output_tensors_to_retain) {
        std::vector<int> granularity;

        int k_lim = 1;
        for (int i = boundary; i < subgraph.size(); i++) {
            Node nd = nodes[subgraph[i]];
            if (nd.is_matmul) {
                k_lim = std::max(k_lim, tensors[nd.inputs[0]].width);
            }
        }
        for (int H = native_granularity[0]; H >= native_granularity[0] / 2; H >>= 1) {
            for (int W = native_granularity[1]; W >= native_granularity[1] / 2; W >>= 1) {
                granularity = {H, W, 1};
                if (cal_memory(granularity) <= fast_memory_capacity) {
                    for (int K = k_lim; K >= 1; K >>= 1) {
                        granularity = {H, W, K};
                        if (cal_memory(granularity) <= fast_memory_capacity) {
                            return granularity;
                        }
                    }
                }
            }
        }
        return granularity;
    }
#undef cal_memory

    void trivial_solution() {
        std::vector<std::vector<int> > subgraphs;
        std::vector<std::vector<int> > granularities;
        std::vector<std::vector<int> > tensors_to_retain;
        std::vector<std::vector<int> > traversal_orders;
        for (int i = 0; i < m; i++) {
            std::vector<int> subgraph;
            std::vector<int> granularity;
            std::vector<int> retain;
            std::vector<int> traversal_order;
            int node_granularity[3];
            if (nodes[i].is_matmul) {
                if (fast_memory_capacity >= native_granularity[0] * native_granularity[1]
                        + (native_granularity[0] + native_granularity[1])) {
                    node_granularity[0] = native_granularity[0];
                    node_granularity[1] = native_granularity[1];
                } else {
                    node_granularity[0] = native_granularity[0] / 2;
                    node_granularity[1] = native_granularity[1] / 2;
                }
                int lim_k = tensors[nodes[i].inputs[0]].width;
                lim_k = std::min(lim_k, (fast_memory_capacity - node_granularity[0] * node_granularity[1])
                                / (node_granularity[0] + node_granularity[1]));
                int k = 1;
                while (k * 2 <= lim_k) {
                    k *= 2;
                }
                subgraph = {i,};
                subgraphs.push_back(subgraph);
                granularity = {node_granularity[0], node_granularity[1], k};
                granularities.push_back(granularity);
            }
            else {
                if (fast_memory_capacity >= native_granularity[0] * native_granularity[1]
                        * (1 + nodes[i].inputs.size())) {
                    node_granularity[0] = native_granularity[0];
                    node_granularity[1] = native_granularity[1];
                } else {
                    node_granularity[0] = native_granularity[0] / 2;
                    node_granularity[1] = native_granularity[1] / 2;
                }
                subgraph = {i,};
                subgraphs.push_back(subgraph);
                granularity = {node_granularity[0], node_granularity[1], 1};
                granularities.push_back(granularity);
            }
            
            // if (tensors[nodes[i].outputs[0]].height * tensors[nodes[i].outputs[0]].width
            //     <= fast_memory_capacity / 2
            //     &&
            //     tensors[nodes[i].outputs[0]].outputs.size() != 0) {
            //         retain.push_back(nodes[i].outputs[0]);
            // }
            tensors_to_retain.push_back(retain);
            // traversal_order = get_traversal_order(subgraph,
            //                                       granularity);
            traversal_orders.push_back(traversal_order);
        }
        check(subgraphs,
              granularities,
              tensors_to_retain,
              traversal_orders,
              myEvaluater);
    }

    int fd_fa(int x, int* fa) {
        if (x != fa[x]) {
            fa[x] = fd_fa(fa[x], fa);
        }
        return fa[x];
    }

    std::vector<std::vector<int> > get_subgraphs() {
        int fa[MAX_M];
        for (int i = 0; i < m; i++) {
            fa[i] = i;
        }
        for (int i = 0; i < n; i++) {
            TensorNode t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (!t.is_cut) {
                int x = fd_fa(t.input, fa);
                for (int j = 0; j < t.outputs.size(); j++) {
                    fa[fd_fa(t.outputs[j], fa)] = x;
                }
            }
        }
        int d_in[MAX_M];
        init(d_in);
        std::vector<std::vector<int> > out;
        out.resize(m);
        for (int i = 0; i < n; i++) {
            TensorNode t = tensors[i];
            if (t.is_input) {
                continue;
            }
            if (t.is_cut) {
                int x = fd_fa(t.input, fa);
                for (int j = 0; j < t.outputs.size(); j++) {
                    int y = fd_fa(t.outputs[j], fa);
                    if (x != y) {
                        out[x].push_back(y);
                        d_in[y]++;
                    }
                }
            }
        }
        int order[MAX_M], cnt = 0;
        std::deque<int> dq;
        for (int i = 0; i < m; i++) {
            if (fa[i] == i) {
                if (d_in[i] == 0) {
                    dq.push_front(i);
                }
            }
        }
        while(dq.size()) {
            int x = dq.front();
            dq.pop_front();
            order[cnt++] = x;
            bool flag = 0;
            for (int i = 0; i < out[x].size(); i++) {
                d_in[out[x][i]]--;
                if (d_in[out[x][i]] == 0) {
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
        std::vector<std::vector<int> > subgraphs;
        for (int i = 0; i < cnt; i++) {
            std::vector<int> subgraph;
            for (int j = 0; j < m; j++) {
                if (fd_fa(j, fa) == order[i]) {
                    subgraph.push_back(j);
                }
            }
            // 由于有额外的 this 键字，所以要写成 lambda 表达式
            sort(subgraph.begin(), subgraph.end(), [this](int x, int y) {
                return this->topo_id[x] < this->topo_id[y];
            });
            subgraphs.push_back(subgraph);
        }
        
        return subgraphs;
    }

    float evaluate_lantency(std::vector<std::vector<int> >& subgraphs,
                            std::vector<std::vector<int> >& granularities,
                            std::vector<std::vector<int> >& tensors_to_retain,
                            std::vector<std::vector<int> >& traversal_orders) {
        std::vector<float> lantency = myEvaluater.evaluate(subgraphs,
                                                            granularities,
                                                            tensors_to_retain,
                                                            traversal_orders);
        if (lantency.size() == 0) {
            return INF_LANTENCY;
        }
        float total_lantency = 0.;
        for (int i = 0; i < lantency.size(); i++) {
            total_lantency += lantency[i];
        }
        return total_lantency;
    }

    float evaluate_lantency_of_cut() {
        std::vector<std::vector<int> > temp_subgraphs = get_subgraphs();
        std::vector<std::vector<int> > subgraphs;
        std::vector<int> subgraph;
        int boundary[MAX_M];
        std::vector<std::vector<int> > d_in, d_out;
        d_in.resize(temp_subgraphs.size());
        d_out.resize(temp_subgraphs.size());
        for (int layer = 0; layer < temp_subgraphs.size(); layer++) {
            subgraph.clear();
            d_in[layer].assign(n, 0);
            d_out[layer].assign(n, 0);
            for (int i = 0; i < temp_subgraphs[layer].size(); i++) {
                Node nd = nodes[temp_subgraphs[layer][i]];
                for (int j = 0; j < nd.outputs.size(); j++) {
                    d_in[layer][nd.outputs[j]]++;
                }
                for (int j = 0; j < nd.inputs.size(); j++) {
                    d_out[layer][nd.inputs[j]]++;
                }
            }

            for (int i = 0; i < temp_subgraphs[layer].size(); i++) {
                if (d_out[layer][nodes[temp_subgraphs[layer][i]].outputs[0]] != 0) {
                    subgraph.push_back(temp_subgraphs[layer][i]);
                }
            }
            boundary[layer] = subgraph.size();
            for (int i = 0; i < temp_subgraphs[layer].size(); i++) {
                if (d_out[layer][nodes[temp_subgraphs[layer][i]].outputs[0]] == 0) {
                    subgraph.push_back(temp_subgraphs[layer][i]);
                }
            }
            subgraphs.push_back(subgraph);
        }

        std::vector<std::vector<int> > granularities;
        std::vector<std::vector<int> > tensors_to_retain;
        std::vector<std::vector<int> > traversal_orders;
        // for (int i = 0; i < subgraphs.size(); i++) {
        //     for (int j = 0; j < subgraphs[i].size(); j++) {
        //         printf("%d ", subgraphs[i][j]);
        //     }
        //     printf("\n");
        // }
        for (int i = 0; i < subgraphs.size(); i++) {
            std::vector<int> retain;
            std::vector<int> granularity;
            std::vector<int> traversal_order;

            retain = get_retain(subgraphs[i], boundary[i]);
            tensors_to_retain.push_back(retain);

            if (i == 0) {
                std::vector<int> empty_tensors_to_retain;
                granularity = get_granularity(subgraphs[i],
                                              boundary[i],
                                              d_in[i],
                                              d_out[i],
                                              empty_tensors_to_retain,
                                              tensors_to_retain[i]);
            } else {
                granularity = get_granularity(subgraphs[i],
                                              boundary[i],
                                              d_in[i],
                                              d_out[i],
                                              tensors_to_retain[i - 1],
                                              tensors_to_retain[i]);
            }
            if (granularity.size() == 0) {
                return INF_LANTENCY;
            }
            granularities.push_back(granularity);
            traversal_order = get_traversal_order(subgraphs[i], granularity);
            traversal_orders.push_back(traversal_order);
        }
        // print(subgraphs,
        //       granularities,
        //       tensors_to_retain,
        //       traversal_orders);
        return evaluate_lantency(subgraphs, granularities, tensors_to_retain, traversal_orders);
    }

    void sa_solution() {
        std::vector<std::vector<int> > subgraphs;
        std::vector<std::vector<int> > granularities;
        std::vector<std::vector<int> > tensors_to_retain;
        std::vector<std::vector<int> > traversal_orders;
        float best_lantency, current_lantency;
        TensorNode best_state[MAX_N];
        for (int i = 0; i < n; i++) {
            best_state[i].is_cut = tensors[i].is_cut = 1;
            best_state[i].is_retained = tensors[i].is_retained = 0;
        }
        best_lantency = evaluate_lantency_of_cut();
        printf("%lf\n", best_lantency);
        float init_temp = 10.0;
        float min_temp = 0.1;
        float cooling_rate = 0.99;
        for (float temp = init_temp; temp >= min_temp;) {
            int x = ran(0, n - 1);
            int flag = ran(0, 2);
            if (!flag) {
                tensors[x].is_cut ^= 1;
            } else {
                tensors[x].is_retained ^= 1;
            }
            current_lantency = evaluate_lantency_of_cut();
            if (current_lantency > INF_LANTENCY / 2) {
                if (!flag) {
                    tensors[x].is_cut ^= 1;
                } else {
                    tensors[x].is_retained ^= 1;
                }
            }
            else {
                if (ran_real(0, 1) < std::exp((best_lantency - current_lantency) / temp)) {
                    if (current_lantency < best_lantency) {
                        for (int i = 0; i < n; i++) {
                            best_state[i].is_cut = tensors[i].is_cut;
                            best_state[i].is_retained = tensors[i].is_retained;
                            best_lantency = current_lantency;
                        }
                    }
                } else {
                    if (!flag) {
                        tensors[x].is_cut ^= 1;
                    } else {
                        tensors[x].is_retained ^= 1;
                    }
                }
                temp *= cooling_rate;
            }
            // printf("%lf %lf\n", current_lantency, best_lantency);
            // printf("!%lf\n", temp);
        }
        printf("%lf\n", best_lantency);
    }
};

#include "json.hpp" 

using json = nlohmann::json;

void check_output(ScheduleEvaluator& myEvaluater) {
    std::ifstream file_out("output.json");
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
    for (const auto& order : j_out["traversal_orders"]) {
        if (order.is_null()) {
            traversal_orders.push_back(std::vector<int>()); 
        }
        else {
            traversal_orders.push_back(order.get<std::vector<int> >());
        }
    }
    std::vector<float> subgraph_latencies = j_out["subgraph_latencies"];

    std::vector<float> lantency = myEvaluater.evaluate(
                                      subgraphs,
                                      granularities,
                                      tensors_to_retain,
                                      traversal_orders
                                  );
    for (int i = 0; i < lantency.size(); i++) {
        printf("%lf\n", lantency[i]);
    }
}

int main() {
    std::ifstream file_in("input.json");
    if (!file_in.is_open()) {
        std::cerr << "Can't open input.json!" << std::endl;
        return 1;
    }
    json j_in;
    file_in >> j_in; 

    std::vector<int> widths = j_in["widths"];
    std::vector<int> heights = j_in["heights"];
    std::vector<std::vector<int> > inputs = j_in["inputs"];
    std::vector<std::vector<int> > outputs = j_in["outputs"];
    std::vector<float> base_costs = j_in["base_costs"];
    std::vector<std::string> op_types = j_in["op_types"];
    float fast_memory_capacity = j_in["fast_memory_capacity"];
    float slow_memory_bandwidth = j_in["slow_memory_bandwidth"];
    std::vector<int> native_granularity = j_in["native_granularity"];

    // std::cout << fast_memory_capacity << std::endl;
    // std::cout << widths[0] << std::endl;
    // std::cout << op_types[0] << std::endl;

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

    

    // check_output(myEvaluater);

    // myComputeGraph.trivial_solution();
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < 10; t++) {
        ComputeGraph myComputeGraph(
            widths,
            heights,
            inputs,
            outputs,
            base_costs,
            op_types,
            fast_memory_capacity,
            slow_memory_bandwidth,
            native_granularity,
            myEvaluater
        );
        myComputeGraph.sa_solution();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "time:" << duration.count() << " ms" << std::endl;
    return 0;
}