// Get the size of unit of each Tensor
Tensor siz[MAX_N];
Tensor layer_siz[MAX_N];
for (int layer = 0; layer < n_layers; layer++) {
    int H = granularities[layer][0];
    int W = granularities[layer][1];
    int K = granularities[layer][2];

    for (int i = boundary[layer]; i < subgraphs[layer].size(); i++) {
        Node nd = nodes[subgraphs[layer][i]];
        for (int j = 0; j < nd.outputs.size(); j++) {
            layer_siz[nd.outputs[j]] = Tensor(H, W);
        }
        if (nd.is_matmul) {
            layer_siz[nd.inputs[0]] = merge(layer_siz[nd.inputs[0]], Tensor(H, K));
            layer_siz[nd.inputs[1]] = merge(layer_siz[nd.inputs[1]], Tensor(K, W));
        }
        else {
            for (int l = 0; l < nd.inputs.size(); l++) {
                layer_siz[nd.inputs[l]] = merge(layer_siz[nd.inputs[l]], Tensor(H, W));
            }
        }
    }

    for (int i = boundary[layer] - 1; i >= 0; i--) {
        Node nd = nodes[subgraphs[layer][i]];
        for (int j = 0; j < nd.outputs.size(); j++) {
            int output_id = nd.outputs[j];
            if (nd.is_matmul) {
                layer_siz[nd.inputs[0]] = merge(layer_siz[nd.inputs[0]], Tensor(layer_siz[output_id].height, tensors[nd.inputs[0]].width));
                layer_siz[nd.inputs[1]] = merge(layer_siz[nd.inputs[1]], Tensor(tensors[nd.inputs[1]].height, layer_siz[output_id].width));
            }
            else {
                for (int l = 0; l < nd.inputs.size(); l++) {
                    layer_siz[nd.inputs[l]] = merge(layer_siz[nd.inputs[l]], layer_siz[output_id]);
                }
            }
        }
    }

    for (int i = 0; i < subgraphs[layer].size(); i++) {
        Node nd = nodes[subgraphs[layer][i]];
        for (int j = 0; j < nd.inputs.size(); j++) {
            siz[nd.inputs[j]] = merge(siz[nd.inputs[j]], layer_siz[nd.inputs[j]]);
        }
        for (int j = 0; j < nd.outputs.size(); j++) {
            siz[nd.outputs[j]] = merge(siz[nd.outputs[j]], layer_siz[nd.outputs[j]]);
        }
    }
}