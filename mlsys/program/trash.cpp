#include<bits/stdc++.h>

#define INF 2000000000
#define MAX_M 100
#define MAX_N 200
#define MAX_SIZ 32
#define init(a); memset(a, 0, sizeof(a));

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
    float siz;
    bool is_load;
    LoadingData(): id(0), height(Range(0, 0)), width(Range(0, 0)), siz(0), is_load(0) {}
    LoadingData(int id, Range height, Range width, float siz, bool is_load) {
        this->id = id;
        this->height = height;
        this->width = width;
        this->siz = siz;
        this->is_load = is_load;
    }
    LoadingData(int id, Range height, Range width, bool is_load) {
        this->id = id;
        this->height = height;
        this->width = width;
        this->siz = (height.r - height.l) * (width.r - width.l);
        this->is_load = is_load;
    }
    LoadingData(int id, int height_l, int height_r, int width_l, int width_r, bool is_load) {
        this->id = id;
        this->height = Range(height_l, height_r);
        this->width = Range(width_l, width_r);
        this->siz = (height_r - height_l) * (width_r - width_l);
        this->is_load = is_load;
    }
};

int gcd(int x,int y) {
    if (y == 0) return x;
    else return gcd(y, x % y);
}

Tensor merge(Tensor x, LoadingData y) {
    // printf("%d %d\n", y.height.r, y.height.l);
    return Tensor(gcd(x.height, y.height.r - y.height.l), gcd(x.width, y.width.r - y.width.l));
}

struct DataPosition {
    int id, h, w;
    float siz;
    bool is_load;
    DataPosition(): id(0), h(0), w(0), siz(0), is_load(0) {}
    DataPosition(int id, int h, int w, float siz, bool is_load) {
        this->id = id;
        this->h = h;
        this->w = w;
        this->siz = siz;
        this->is_load = is_load;
    }
    bool operator<(const DataPosition& other) const {
        return id == other.id ? (h == other.h ? w < other.w : h < other.h) : id < other.id;
    }
};

class Belady {
private:
    std::vector<int> tim[MAX_N][MAX_SIZ][MAX_SIZ];
    int pos[MAX_N][MAX_SIZ][MAX_SIZ];

    int next_tim(DataPosition x) {
        return tim[x.id][x.w][x.h].size() > pos[x.id][x.w][x.h]
             ? tim[x.id][x.w][x.h][pos[x.id][x.w][x.h]]
             : INF;
    }

    std::priority_queue<std::pair<int, DataPosition> > q;
    std::set<DataPosition> st;
    float fast_memory_capacity;
    float used_memory_capacity;

public:
    Belady (std::vector<LoadingData>& memory_trace) {
        memset(pos, 0, sizeof(pos));
        for (int i = 0; i < memory_trace.size(); i++) {
            LoadingData dat = memory_trace[i];
            for (int h = dat.height.l; h < dat.height.r; h++) {
                for (int w = dat.width.l; w < dat.width.r; w++) {
                    tim[dat.id][h][w].push_back(i);
                }
            }
        }
    }

    float reset(float fast_memory_capacity) {
        this->fast_memory_capacity = fast_memory_capacity;
        used_memory_capacity = 0;
        st.clear();
        float cost = 0.;
        while(q.size()) {
            int dat_tim = q.top().first;
            DataPosition dat = q.top().second;
            q.pop();
            if (dat_tim == next_tim(dat)) {
                if (!dat.is_load) {
                    cost += dat.siz;
                }
            }
        }
        return cost;
    }

    void pop(DataPosition x) {
        pos[x.id][x.w][x.h]++;
    }

    float ins(DataPosition x) {
        if (st.count(x)) {
            pos[x.id][x.w][x.h]++;
            q.push(std::make_pair(next_tim(x), x));
            return 0;
        }
        float cost = 0.;
        if (x.is_load) cost += x.siz;
        while (used_memory_capacity + x.siz > fast_memory_capacity) {
            int dat_tim = q.top().first;
            DataPosition dat = q.top().second;
            q.pop();
            if (dat_tim == next_tim(dat)) {
                st.erase(dat);
                used_memory_capacity -= dat.siz;
                if (!dat.is_load) {
                    cost += dat.siz;
                }
            }
        }
        used_memory_capacity += x.siz;
        pos[x.id][x.w][x.h]++;
        st.insert(x);
        q.push(std::make_pair(next_tim(x), x));
        return cost;
    }
};

class ScheduleEvaluator {
private:
    int n, m;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
    float fast_memory_capacity;
    float slow_memory_bandwidth;
    std::vector<int> native_granularity;
public:
    ScheduleEvaluator (
        const std::vector<int>& widths,
        const std::vector<int>& heights,
        const std::vector<std::vector<int> >& inputs,
        const std::vector<std::vector<int> >& outputs,
        const std::vector<float>& base_costs,
        const std::vector<std::string>& op_types,
        const float fast_memory_capacity,
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
        std::vector<std::vector<int> >& subgraphs,
        std::vector<std::vector<int> >& granularities,
        std::vector<std::vector<int> >& tensors_to_retain,
        std::vector<std::vector<int> >& traversal_orders
    ) {
        printf("!\n");
        fflush(stdout);

        // Get necesssary data
        auto& n = this->n;
        auto& m = this->m;
        auto& tensors = this->tensors;
        auto& nodes = this->nodes;
        auto& fast_memory_capacity = this->fast_memory_capacity;
        auto& slow_memory_bandwidth = this->slow_memory_bandwidth;
        auto& native_granularity = this->native_granularity;
        
        int n_layers = subgraphs.size();
        assert(n_layers == granularities.size());
        assert(n_layers == tensors_to_retain.size());
        assert(n_layers == traversal_orders.size());

        int d_in[MAX_N][MAX_N];
        int d_out[MAX_N][MAX_N];
        for (int i = 0; i < subgraphs.size(); i++) {
            init(d_in[i]);
            init(d_out[i]);
            for (int j = 0; j < subgraphs[i].size(); j++) {
                Node nd = nodes[subgraphs[i][j]];
                for (int k = 0; k < nd.outputs.size(); k++) {
                    d_in[i][nd.outputs[k]]++;
                }
                for (int k = 0; k < nd.inputs.size(); k++) {
                    d_out[i][nd.inputs[k]]++;
                }
            }
        }

        printf("!\n");

        // Check the legitimacy
        bool used_op[MAX_M];
        bool is_computed[MAX_N];
        for (int i = 0; i < subgraphs.size(); i++) {
            for (int j = 0; j < subgraphs[i].size(); j++) {
                used_op[subgraphs[i][j]] = 1;
            }
        }
        for (int i = 0; i < m; i++) {
            assert(used_op[i]);
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
                        assert(ephemeral_computed[nd.inputs[k]]);
                    }
                    else {
                        assert(is_computed[nd.inputs[k]]);
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

        printf("!\n");

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

        printf("!\n");

        // OOM check
        std::vector<Tensor> single_compute_size[MAX_N];
        int retain_memory, temp_memory;
        for (int layer = 0; layer < n_layers; layer++) {
            retain_memory = 0;
            for (int i = std::max(layer - 1, 0); i < layer; i++) {
                for (int j = 0; j < tensors_to_retain[i].size(); j++) {
                    retain_memory += tensors[tensors_to_retain[i][j]].height * tensors[tensors_to_retain[i][j]].width;
                }
            }
            int H = granularities[layer][0];
            int W = granularities[layer][1];
            int K = granularities[layer][2];
            for (int i = 0; i < n; i++) {
                single_compute_size[i].clear();
            }

            for (int i = subgraphs[layer].size() - 1; i >= boundary[layer]; i--) {
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
                                temp_memory += cal_h * cal_k;
                            }
                            if (d_in[layer][nd.inputs[1]]) {
                                single_compute_size[nd.inputs[1]].push_back(Tensor(cal_k, cal_w));
                            }
                            else {
                                temp_memory += cal_k * cal_w;
                            }
                        }
                        else {
                            for (int l = 0; l < nd.inputs.size(); l++) {
                                if (d_in[layer][nd.inputs[l]]) {
                                    single_compute_size[nd.inputs[l]].push_back(Tensor(dat.height, dat.width));
                                }
                                else {
                                    temp_memory += dat.height * dat.width;
                                }
                            }
                        }
                        assert(retain_memory + temp_memory <= fast_memory_capacity);
                    }
                }
            }
        }

        printf("!\n");

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
        for (int layer = 0; layer < n_layers; layer++) {
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
                                        requests[output_id].push_back(LoadingData(output_id, hh * H, hh * H + H, ww * W, ww * W + W, 1));
                                    } else {
                                        requests[output_id].push_back(LoadingData(output_id, h * H, h * H + H, w * W, w * W + W, 1));
                                    }
                                }

                                while(requests[output_id].size()) {
                                    turn_compute_lantency += nd.cost;
                                    LoadingData dat = requests[output_id].back();
                                    requests[output_id].pop_back();
                                    if (nd.is_matmul) {
                                        if (i < boundary[layer] || k < tensors[nd.inputs[0]].width / K) {
                                            Range cal_h = dat.height;
                                            Range cal_w = dat.width;
                                            Range cal_k = (i >= boundary[layer] ? Range(k * K, k * K + K) : Range(0, tensors[nd.inputs[0]].width));
                                            assert(cal_h.r - cal_h.l <= native_granularity[0] && cal_w.r - cal_w.l <= native_granularity[1]);
                                            if (d_in[layer][nd.inputs[0]]) {
                                                requests[nd.inputs[0]].push_back(LoadingData(nd.inputs[0], cal_h, cal_k, 1));
                                            }
                                            else {
                                                layer_trace.push_back(LoadingData(nd.inputs[0], cal_h, cal_k, 1));
                                            }
                                            if (d_in[layer][nd.inputs[1]]) {
                                                requests[nd.inputs[1]].push_back(LoadingData(nd.inputs[1], cal_k, cal_w, 1));
                                            }
                                            else {
                                                layer_trace.push_back(LoadingData(nd.inputs[1], cal_k, cal_w, 1));
                                            }
                                        }
                                    }
                                    else {
                                        assert(dat.height.r - dat.height.l <= native_granularity[0] && dat.width.r - dat.width.l <= native_granularity[1]);
                                        for (int l = 0; l < nd.inputs.size(); l++) {
                                            if (d_in[layer][nd.inputs[l]] > 0) {
                                                requests[nd.inputs[l]].push_back(LoadingData(nd.inputs[l], dat.height, dat.width, 1));
                                            }
                                            else {
                                                layer_trace.push_back(LoadingData(nd.inputs[l], dat.height, dat.width, 1));
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
            printf("!%d %d %d %d %d\n", memory_trace[i].id, memory_trace[i].height.l, memory_trace[i].height.r, memory_trace[i].width.l, memory_trace[i].width.r);
        }

        printf("!\n");

        // Belady
        Belady myBelady(memory_trace);
        bool retained_tensor[MAX_N];
        int used_memory_capacity;
        float turn_memory_lantency;
        float layer_lantency;
        std::vector<float> lantency;
        for (int layer = 0; layer < subgraphs.size(); layer++) {
            int layer_l = (layer == 0 ? 0 : layer_boundary[layer - 1]);
            int layer_r = layer_boundary[layer];
            used_memory_capacity = 0;
            if (layer > 0) {
                for (int i = 0; i < tensors_to_retain[layer - 1].size(); i++) {
                    retained_tensor[tensors_to_retain[layer - 1][i]] = 1;
                    used_memory_capacity += tensors[tensors_to_retain[layer - 1][i]].width
                        * tensors[tensors_to_retain[layer - 1][i]].height;
                }
            }
            for (int i = 0; i < tensors_to_retain[layer].size(); i++) {
                retained_tensor[tensors_to_retain[layer][i]] = 1;
                used_memory_capacity += tensors[tensors_to_retain[layer - 1][i]].width
                    * tensors[tensors_to_retain[layer][i]].height;
            }
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
                                myBelady.pop(DataPosition(dat.id, h, w, dat.siz, dat.is_load));
                            }
                            else {
                                turn_memory_lantency += myBelady.ins(DataPosition(dat.id, h, w, siz[dat.id].height * siz[dat.id].width, dat.is_load));
                                printf("%d %d %d %lf\n", dat.id, h, w, turn_memory_lantency);
                            }
                        }
                    }
                }
                if (i == layer_r - 1) {
                    turn_memory_lantency += myBelady.reset(0);
                }
                layer_lantency += std::max(compute_lantency[i], turn_memory_lantency / slow_memory_bandwidth);
                // printf("!%lf", turn_memory_lantency);
            }
            lantency.push_back(layer_lantency);
            if (layer > 0) {
                for (int i = 0; i < tensors_to_retain[layer - 1].size(); i++) {
                    retained_tensor[tensors_to_retain[layer - 1][i]] = 0;
                }
            }
            for (int i = 0; i < tensors_to_retain[layer].size(); i++) {
                retained_tensor[tensors_to_retain[layer][i]] = 0;
            }
        }

        return lantency;
    }
};

#include "json.hpp" 

using json = nlohmann::json;

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


    std::ifstream file_out("output.json");
    if (!file_out.is_open()) {
        std::cerr << "Can't open output.json!" << std::endl;
        return 1;
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
    return 0;
}

    float cal_lantency(std::vector<int>& subgraph, std::vector<int>& granularity) {
        int h[MAX_N][2], w[MAX_N][2];
        float lantency = 0.;
        init(h);init(w);
        for (int i = subgraph.size() - 1; i >= 0; i--) {
            Node nd = nodes[subgraph[i]];
            if (i == subgraph.size() - 1) {
                if (granularity[0] > native_granularity[0] || granularity[1] > native_granularity[1]) {
                    printf("Exceed native_granularity!\n");
                    return INF_LANTENCY;
                }
                if (nd.is_matmul) {
                    h[nd.inputs[0]][0] = h[nd.inputs[0]][1] = granularity[0];
                    w[nd.inputs[0]][0] = w[nd.inputs[0]][1] = granularity[2];
                    h[nd.inputs[1]][0] = h[nd.inputs[1]][1] = granularity[2];
                    w[nd.inputs[1]][0] = w[nd.inputs[1]][1] = granularity[1];
                    lantency += (float)nd.cost / granularity[2];
                } else {
                    for (int j = 0; j < nd.inputs.size(); j++) {
                        h[nd.inputs[j]][0] = h[nd.inputs[j]][1] = granularity[0];
                        w[nd.inputs[j]][0] = w[nd.inputs[j]][1] = granularity[1];
                    }
                    lantency += (float)nd.cost;
                }
            } else {
                if (h[nd.outputs[0]][0] > native_granularity[0] || w[nd.outputs[0]][1] > native_granularity[1]) {
                    printf("Exceed native_granularity!\n");
                    return INF_LANTENCY;
                }
                if (h[nd.outputs[0]][0] != h[nd.outputs[0]][1] && w[nd.outputs[0]][0] != w[nd.outputs[0]][1]) {
                    lantency += (float)nd.cost * 2;
                } else {
                    lantency += (float)nd.cost;
                }
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
                } else {
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
        return lantency;
    }

    bool upd_granularity(int layer) {
        std::vector<int> granularity, best_granularity;
        std::vector<int> best_traversal_order;
        float lantency, best_lantency = INF_LANTENCY;

        int k_lim_r = 1, k_lim_l = 1;
        Node nd = nodes[subgraphs[layer].back()];
        if (nd.is_matmul) {
            k_lim_r = std::max(k_lim_r, tensors[nd.inputs[0]].width);
        }
        std::vector<int> temp_granularity = granularities[layer];
        for (int H = native_granularity[0]; H >= native_granularity[0] / 2; H >>= 1) {
            for (int W = native_granularity[1]; W >= native_granularity[1] / 2; W >>= 1) {
                granularity = {H, W, 1};
                if (cal_memory(granularity, layer) <= fast_memory_capacity) {
                    for (int K = k_lim_r; K >= 1; K >>= 1) {
                        granularity = {H, W, K};
                        if (cal_memory(granularity, layer) <= fast_memory_capacity) {
                            granularities[layer] = granularity;
                            lantency = upd_traversal_order(layer);
                            if (lantency < best_lantency) {
                                best_lantency = lantency;
                                best_granularity = granularity;
                                best_traversal_order = traversal_orders[layer];
                            }
                            break; // delete
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