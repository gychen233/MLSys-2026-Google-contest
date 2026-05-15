评估器

### 数据搬运

利用 vector 进行高效搬运，并且把一部分数据永久化，从而支持多次查询。

### 合法性检查

依次对于每一个 sub_graph 进行合法性检查，看看是不是每一个输出都已经提前算出。

### lantency 计算与 OOM 判定

#### 展平

在每个子图里面，先找到 output 与 input，然后反向地传播。我们直接记录每一个操作所需的内存与输出的内存，形式是（Tensor_id，行范围，列范围）（分块后）。

#### Belady 最优替换算法

依照上面一步得到的所有步骤，处理内存，可以直接暴力用 set 去处理。

过程中计算每一步的 lantency 与 OOM。

小心 OOM 的判定方法。

# 注意最后的输出不能 retain，不然我的检测器会出 bug。

# 已知有很多 bug，尤其是多输出的时候，很多情形没有处理

# 还有很多实现过于繁复，需要重构优化



Q20 Q28



我想询问的是“incompatible”的定义，假如两个输出矩阵大小分别是 256*256 和 128*256，我可不可以以 128*128 粒度划分，在超出其中一个矩阵边界的时候对他的执行做静默处理？以及，我可不可以两个输出的操作，一个是matmul，一个是pointwise？



I would like to clarify the definition of 'incompatible' when grouping operations. Suppose two operations in a subgraph produce output matrices of sizes 256x256 and 128x256, respectively. Can I use an execution granularity of 128x128 and simply mask/silently ignore the execution when the spatial grid exceeds the boundaries of the smaller matrix? Additionally, is it allowed to group two operations  in the same subgraph if one is a MatMul and the other is a Pointwise operation?



I fully thank the organizers for addressing our comments, but I still have a lot of confusion.

As an extension of Q20 Q28.

I would like to clarify the definition of 'incompatible' when grouping operations. 

Suppose two operations in a subgraph produce output matrices of sizes 256x256 and 128x256, respectively. Can I use an execution granularity of 128x128 and simply mask/silently **ignore** the execution when the spatial grid exceeds the boundaries of the smaller matrix? 

Additionally, is it allowed for a single subgraph to have multiple **output operations** of different types—specifically, one being a MatMul and the other a Pointwise operation?



我想再进一步询问关于临时内存的问题，处理临时内存是采取最优策略吗？最优策略具体来说是 Belady 方法吗？为了到达最优策略，在释放一块空间的时候必须释放整块数据，还是可以释放一部分？如果出现一个 matmul 的输入是同一个Tensor，那么临时内存存取的是这一个 Tensor一行和一列，可以认为重叠部分只占一份内存？以及如果我们最后lantency计算错误，会被判为零分吗？



As an extension of Q15 Q37.

I would like to ask a few more questions regarding **Implicit Reuse**. 

Does 'implicit reuse' only trigger between adjacent executions, or can I **arbitrarily** retain data in the fast memory to achieve **better implicit reuse**?

Does the latency calculation assume an **optimal** memory scheduling strategy by default?

Specifically, does it use **Bélády's optimal algorithm**? 

To achieve this strategy, when freeing up space, must we evict **an entire data block** at once, or can we evict **partial slices**? 

Additionally, if a MatMul operation takes the same Tensor as both of its inputs, the fast memory needs to load **both a horizontal strip and a vertical strip** of that Tensor. Can we assume that the overlapping intersection of these strips only consumes **a single copy of capacity** within the fast memory limit?

 Finally, if our calculated `subgraph_latencies` are incorrect, will our submission receive a score of zero?



OOM 判断 $\checkmark$



要作为输出的 Tensor，retain 了，没有运算输出的 lantency $\checkmark$



没有检查 retain 的张量是否一定是输出



subgraph 之间的顺序？ $\checkmark$



d_in d_out 处理那一块，默认了只有一个输出。



默认了所有输出的维度大小一致。



一个优化思路是，我们不一定是切掉一整个 Tensor，而只是切掉它的入边，然后出边切一部分。以及关于子图序，可以根据这个优化与验证。



调换 granularity 顺序或许有用。



traversal order 应该有更优形式，针对由于融合，所以不平衡的算子。



对于多个输出，也许有更好的处理。



针对一些定态数组，需要变为 vector，对于各个细节进行优化，提升速度。



完善中途拦截机制，保证速度。



granularity_order 的取法太随机，在 subgraph 平移的时候会崩溃。



模拟退火未必是最优的方法，也许我们可以先模拟退火产生一个子图分割方案，然后根据方案针对地模拟退火随机其他参量。



resize  assign



retained 的如果是多个算子的输入，那么可能在某一时刻需要读到 slow_memory 里面。



改为long long与double。



json.hpp可能有问题



要在linux环境编译一次







I'm a little confused by the concept of granularity. I appreciate you taking the time to answer these few questions for me.

Q1: Can **granularity[0]** be greater than **native_granularity[0]** ? ( And can **granularity[1]** be greater than **native_granularity[1]** ? )

![img](https://private-user-images.githubusercontent.com/256737059/577256053-a77cd59b-0272-49d7-926a-57f0181cb3fa.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzYwNzEzMjMsIm5iZiI6MTc3NjA3MTAyMywicGF0aCI6Ii8yNTY3MzcwNTkvNTc3MjU2MDUzLWE3N2NkNTliLTAyNzItNDlkNy05MjZhLTU3ZjAxODFjYjNmYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNDEzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDQxM1QwOTAzNDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mOGI3MDBhMjYzOTg5OWE1YzUwOTU4NGEzNDY0MjYwOTMwZjJkNjhjNGIyYjJhZDc5ZTAzOGM3MGZkZDJjMjA3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9aW1hZ2UlMkZwbmcifQ.A8JuekQSmX6TqF6_aA-18v6brU_6RKDHOPI-j4fGnUE)

Q2: For this eample, if memory is sufficient and native_granularity is [128, 128], can I fuse all the nodes and use granularity [128, 128, 32] ?

Q3: If it is feasible, Is the computing time for a single turn calculated as $2000 \cdot \frac{256}{128} + 2000 \cdot \frac{32}{256} = 4250$?



The answer in  is very reasonable. I comprehend that the granularity is apply to all the operators in the current subgraph. I want to clarify more details about it.

Q1: Is the single execution size limitation of an operation determined by the native_granularity or the granularity? For example, if the native_granularity is [128, 128] and I set the granularity to [64, 64, 64], can the single execution size of an operator still be 128?

Q2: Is the granularity[2] also uniformly enforced across all operations within the subgraph? For instance, in example #74, is op0 required to use this specific reduction depth and execute eight times to get the output?



