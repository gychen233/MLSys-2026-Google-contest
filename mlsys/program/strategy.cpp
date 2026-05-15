void trivial_solution(ScheduleEvaluator myEvaluater) {
        std::vector<std::vector<int> > subgraphs;
        std::vector<std::vector<int> > granularities;
        std::vector<std::vector<int> > tensors_to_retain;
        std::vector<std::vector<int> > traversal_orders;
        for (int i = 0; i < m; i++) {
            std::vector<int> subgraph;
            std::vector<int> granularity;
            std::vector<int> retain;
            std::vector<int> traversal_order;
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
                while (k * 2 < lim_k) {
                    k *= 2;
                }
                std::vector<int> subgraph = {i,};
                subgraphs.push_back(subgraph);
                std::vector<int> granularity = {node_granularity[0], node_granularity[1], k};
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
                std::vector<int> subgraph = {i,};
                subgraphs.push_back(subgraph);
                std::vector<int> granularity = {node_granularity[0], node_granularity[1], 1};
                granularities.push_back(granularity);
            }
            
            tensors_to_retain.push_back(retain);
            traversal_order = get_traversal_order(tensors[nodes[i].outputs[0]].height / granularity[0],
                                                  tensors[nodes[i].outputs[0]].width / granularity[1]);
            printf("%d %d\n", tensors[nodes[i].outputs[0]].height / granularity[0], tensors[nodes[i].outputs[0]].width / granularity[1]);
            for (int i = 0; i < traversal_order.size(); i++) {
                printf("%d ", traversal_order[i]);
            }
            printf("\n");
            traversal_orders.push_back(traversal_order);
        }
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