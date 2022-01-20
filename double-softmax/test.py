import torch
import torch.nn as nn
import time
import math
total_start = time.time()
start = time.time()
q = 16 * torch.rand(1, 100, 50) - 8
k = 16 * torch.rand(1, 100, 50) - 8
v = 16 * torch.rand(1, 100, 50) - 8
end = time.time()
print("产生矩阵所需要的时间：%.8f秒" % (end - start))
start1 = time.time()
attn_output_weights = torch.bmm(q, k.transpose(1, 2))
attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
attention_output = torch.bmm(attn_output_weights, v)
print(attention_output.size())
end1 = time.time()
# print("标准attention函数运行时间：%.8f秒" % (end1 - start1))
start2 = time.time()
q = torch.softmax(q, dim=-1)
k = torch.softmax(k, dim=1)
attn_output_weights = torch.bmm(k.transpose(1, 2), v)
double_softmax_output = torch.bmm(q, attn_output_weights)
end2 = time.time()
print("double-softmax函数运行时间：%.8f秒" % (end2 - start2))
print(double_softmax_output.size())
start3 = time.time()
m = nn.ELU()
q = m(q) + 1
k = m(k) + 1
q = q.transpose(1, 2)
q_sum = torch.sum(q, 1)
q = q / q_sum
q = q.transpose(1, 2)
k_sum = torch.sum(k, 1)
k = k / k_sum
attn_output_weights = torch.bmm(k.transpose(1, 2), v)
kernel_output = torch.bmm(q, attn_output_weights)
end3 = time.time()
print("kernel函数运行时间：%.8f秒" % (end3 - start3))
print(kernel_output.size())
start4 = time.time()
q = torch.nn.functional.normalize(q)+1
k = torch.nn.functional.normalize(k)+1
q = q.transpose(1, 2)
q_sum = torch.sum(q, 1)
q = q / q_sum
q = q.transpose(1, 2)
k_sum = torch.sum(k, 1)
k = k / k_sum
attn_output_weights = torch.bmm(k.transpose(1, 2), v)
Taylor_norm_output = torch.bmm(q, attn_output_weights)
end4 = time.time()
print("Taylor-norm函数运行时间：%.8f秒" % (end4 - start4))
print(Taylor_norm_output.size())
total_end = time.time()
print("总共运行需要的时间：%.8f秒" % (total_end - total_start))

attn_output11 = double_softmax_output - attention_output  # 三种方法和标准attention对应元素相减
attn_output22 = kernel_output - attention_output
attn_output33 = Taylor_norm_output - attention_output

double_softmax_abs = attn_output11 * attn_output11  # 三种方法和标准attention对应元素相减得到的矩阵所有元素取平方
kernel_abs = attn_output22 * attn_output22
Taylor_norm_abs = attn_output33 * attn_output33
# print(double_softmax_abs)
# print(kernel_abs)
# print(Taylor_norm_abs)

double_softmax_abs_sum = double_softmax_abs.sum().item()  # 三种方法和标准attention对应元素相减得到的矩阵所有元素取平方求和
kernel_abs_sum = kernel_abs.sum().item()
Taylor_norm_abs_sum = Taylor_norm_abs.sum().item()

# print(double_softmax_abs_sum)
# print(kernel_abs_sum)
# print(Taylor_norm_abs_sum)

attention_output_abs = attention_output * attention_output  # 求标准attention的元素的平方和
attention_output_abs_sum = attention_output_abs.sum().item()

double_softmax_abs_sum_sqrt = math.sqrt(double_softmax_abs_sum)  # 三种方法和标准attention对应元素相减得到的矩阵所有元素取平方求和开根号
kernel_abs_sum_sqrt = math.sqrt(kernel_abs_sum)
Taylor_norm_abs_sum_sqrt = math.sqrt(Taylor_norm_abs_sum)
attention_output_abs_sum_sqrt = math.sqrt(attention_output_abs_sum)
print(double_softmax_abs_sum_sqrt)
print(kernel_abs_sum_sqrt)
print(Taylor_norm_abs_sum_sqrt)
print("-----------------------------")


print(double_softmax_abs_sum_sqrt / attention_output_abs_sum_sqrt)  # 三种方法的平方求和开根号 /  求标准attention的元素的平方和开根号
print(kernel_abs_sum_sqrt / attention_output_abs_sum_sqrt)
print(Taylor_norm_abs_sum_sqrt / attention_output_abs_sum_sqrt)

