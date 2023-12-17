import torch
from utils import HuffmanCoding
import numpy as np
from configparser import ConfigParser

# MSB
# e.g. [0, 1, 1, 1] looks like 0111=7
# 将MSB形式的位列表转换成整数
def msb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits[::-1]):
        res += bit * (2 ** i)
    return res

# 将整数转换成MSB形式的位列表
def msb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in strlist]

# lsb
# e.g. [0, 1, 1, 1] looks like 1110=14
# 将LSB形式的位列表转换成整数
def lsb_bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res

# 将整数转换成LSB形式的位列表
def lsb_int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


# 用于计算两个位列表的相同前缀长度
def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break
    return i

# 负责寻找一个排序后列表中离目标值最近的一组索引
def near(alist, anum):
    up = len(alist) - 1
    if up == 0:
        return 0
    bottom = 0
    while up - bottom > 1:
        index = int((up + bottom) / 2)
        if alist[index] < anum:
            up = index
        elif alist[index] > anum:
            bottom = index
        else:
            return index
    if up - bottom == 1:
        if alist[bottom] - anum < anum - up:
            index = bottom
        else:
            index = up
    return index

# 负责寻找一个排序后的列表中离目标值最近的一组索引
def find_nearest_list(prob, delta,):
    diff = (np.array(prob) - delta)
    tmp_idx = np.argmin(diff**2)
    if prob[tmp_idx] < delta:
        return_list = [tmp_idx]
        if tmp_idx == len(prob) -1:
            pass
        else:
            tmp_sum = prob[tmp_idx]
            for i in range(tmp_idx+1, len(prob)-1):
                if delta>(tmp_sum + prob[i]):
                    tmp_sum += prob[i]
                    return_list.append(i)
        return return_list
    elif tmp_idx >= len(prob)-2:
        return [tmp_idx]
    else:
        new_idx = tmp_idx + 1
        idx = [new_idx]
        idx += find_nearest_list(prob[new_idx+1:], delta-prob[new_idx])
        for i in range(1, len(idx)):
            idx[i] += new_idx+1
        # return idx
        if (delta-np.sum(np.array(prob)[idx]))**2 > diff[tmp_idx]**2:
            return [tmp_idx]
        else:
            return idx

# 负责寻找一个排序后的列表中离目标值最近的索引
def find_nearest(prob, delta,):
    diff = (np.array(prob) - delta)
    tmp_idx = np.argmin(diff**2)
    if prob[tmp_idx] < delta:
        return tmp_idx
    elif tmp_idx >= len(prob)-2:
        return tmp_idx
    else:
        new_idx = tmp_idx+1
        idx = find_nearest(prob[new_idx+1:], delta-prob[new_idx])
        idx += new_idx+1
        return idx

# 根据概率将索引分为两组
def grouping(prob):
    # prob = prob/prob.sum()
    prob, indices = prob.sort(descending=True)
    prob = prob.tolist()
    indices = indices.tolist()
    mean = 0.5
    # initialize
    groups = [[0,[]],[0,[]]]
    if prob[0] > mean:
        groups[0] = [prob[0], [indices[0]]]
        groups[1] = [1-prob[0], indices[1:]]
    else:
        groups[0] = [prob[0], [indices[0]]]
        del prob[0]
        del indices[0]
        delta = mean - groups[0][0]
        # while delta >= error_threshold and abs(delta-prob[-1]) > 1/2 * delta:
        # while prob[-1] < 2 * delta:
        #     idx = find_nearest(prob, delta)
        #     groups[0][0] += prob[idx]
        #     groups[0][1].append(indices[idx])
        #     del prob[idx]
        #     del indices[idx]
        while prob[-1] < 2*delta:
            idx_list = find_nearest_list(prob, delta)
            sorted_idx_list = np.sort(idx_list)[::-1]
            for idx in sorted_idx_list:
                groups[0][0] += prob[idx]
                groups[0][1].append(indices[idx])
                del prob[idx]
                del indices[idx]
            delta = mean - groups[0][0]
            # delta = abs(mean - groups[0][0])
        groups[1][0] = 1 - groups[0][0]
        groups[1][1] = indices
    return groups

# 隐写算法的解码器，使用固定长度编码
def FLC_decoder(prob, prev, bit, **kwargs):
    prob, indices = prob.sort(descending=True)
    topk = 2 ** bit
    # prob = prob[:topk]
    indices = indices[:topk]
    if prev in indices:
        msb_number = (indices - prev).abs().argmin().item()
        bits = msb_int2bits(msb_number, bit)
        bits = "".join([str(_) for _ in bits])
        return bits
    else:
        return ""

# 隐写算法的解码器，使用霍夫曼编码
def HC_decoder(prob, prev,  bit, **kwargs):
    prob, indices = prob.sort(descending=True)
    prob = prob[:2 ** bit]
    indices = indices[:2 ** bit]
    if prev in indices:
        node = int((indices == prev).nonzero()[0][0])
        prob_dict = {i: float(p) for i, p in enumerate(prob)}
        hf = HuffmanCoding()
        hf.make_heap(prob_dict)
        hf.merge_nodes()
        hf.make_codes()
        return hf.codes[node]
    else:
        return ""

# 隐写算法解码器，使用自适应编码
def AC_decoder(prob, prev, cur_interval, precision=52, **kwargs):
    prob, indices = prob.sort(descending=True)
    # prob = prob[:2 ** Generation_Configs.bit]
    # indices = indices[:2 ** Generation_Configs.bit]
    # arithmetic coding
    cur_int_range = cur_interval[1] - cur_interval[0]  # 区间的大小  2^26
    cur_threshold = 1 / cur_int_range  # 每个区间多大
    if prob[-1] < cur_threshold:
        k = max(2, (prob < cur_threshold).nonzero()[0].item())
        prob = prob[:k]
        indices = indices[:k]


    prob = prob / prob.sum()  # 截断后线性归一化
    prob = prob.double()
    prob *= cur_int_range  # 概率转换为多少个区间
    prob = prob.round().long()  # 四舍五入取整，区间数描述的概率

    cum_probs = prob.cumsum(0)  # 前面所有项的和的序列区间数描述的分布函数，按理讲最后应该与区间数相同
    overfill_index = (cum_probs > cur_int_range).nonzero()
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]  #去掉最后一个概率

    if prev in indices and prev not in indices[overfill_index]:
        cum_probs += cur_int_range - cum_probs[-1]  # 分布函数加到和区间数相等，区间数表示的分布函数
        cum_probs += cur_interval[0]  # 分布函数的第一项从左区间开始
        selection = (indices==prev).nonzero()[0].item()

        new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[
            0]  # 新的左区间 如果选了第一个单词（selection=0）就代表不需要动区间的左边界
        new_int_top = cum_probs[selection]

        new_int_bottom_bits_inc = list(msb_int2bits(new_int_bottom, precision))  # 二进制的下边界
        new_int_top_bits_inc = list(msb_int2bits(new_int_top - 1, precision))
        # new_int_bottom_bits_inc = list(reversed(lsb_int2bits(new_int_bottom, precision)))  # 二进制的下边界
        # new_int_top_bits_inc = list(reversed(lsb_int2bits(new_int_top - 1, precision)))  # 二进制的上边界

        num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
        bits = "".join([str(b) for b in new_int_bottom_bits_inc[:num_bits_encoded]])
        new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded  # 新二进制区间
        new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

        cur_interval[0] = msb_bits2int(new_int_bottom_bits)  # 新的区间
        cur_interval[1] = msb_bits2int(new_int_top_bits) + 1  # +1 here because upper bound is exclusive
        return cur_interval, bits
    else:
        return cur_interval, ""

# 隐写算法解码器，使用动态规划编码
def ADG_decoder(prob, prev, **kwargs):
    device = prob.device
    prob, indices = prob.sort(descending=True)
    # start recursion
    bit_tmp = 0
    extract_bits = ""
    while prob[0] <= 0.5:
        # embedding bit
        bit = 1
        while (1 / 2 ** (bit + 1)) > prob[0]:
            bit += 1
        mean = 1 / 2 ** bit
        # dp
        prob = prob.tolist()
        indices = indices.tolist()
        result = []
        for i in range(2 ** bit):
            result.append([[], []])
        for i in range(2 ** bit - 1):
            result[i][0].append(prob[0])
            result[i][1].append(indices[0])
            del (prob[0])
            del (indices[0])
            while sum(result[i][0]) < mean:
                delta = mean - sum(result[i][0])
                index = near(prob, delta)
                if prob[index] - delta < delta:
                    result[i][0].append(prob[index])
                    result[i][1].append(indices[index])
                    del (prob[index])
                    del (indices[index])
                else:
                    break
            mean = sum(prob) / (2 ** bit - i - 1)
        result[2 ** bit - 1][0].extend(prob)
        result[2 ** bit - 1][1].extend(indices)
        bit_embed = ""
        for int_embed, result_tmp in enumerate(result):
            if prev in result_tmp[1]:
                bit_embed = "".join([str(b) for b in msb_int2bits(int_embed, bit)])
                break
        prob = torch.FloatTensor(result[int_embed][0]).to(device)
        indices = torch.LongTensor(result[int_embed][1]).to(device)
        prob = prob / prob.sum()
        prob, _ = prob.sort(descending=True)
        indices = indices[_]
        extract_bits += bit_embed
    return extract_bits

# 这是ADG解码器的变体版本，采用不同的策略
def ADG_V2_decoder(prob, prev, epsilon, max_bit, **kwargs):
    mean = 0.5
    # ori_prob = prob
    prob, indices = prob.sort(descending=True)
    # prob_sum = prob.sum()
    acc_prob_sum = 1
    # start recursion
    bit_tmp = 0
    return_bits = ""
    # epsilon = Generation_Configs.epsilon
    groups = grouping(prob)
    while (abs(groups[0][0] - mean) <= epsilon * (2 ** bit_tmp)) and abs(
            groups[0][0] - mean) < mean and bit_tmp < max_bit:
        if prev in indices[groups[0][1]]:
            prob = prob[groups[0][1]]
            indices = indices[groups[0][1]]
            return_bits += "0"
        else:
            prob = prob[groups[1][1]]
            indices = indices[groups[1][1]]
            return_bits += "1"
        prob_sum = prob.sum()
        acc_prob_sum *= prob_sum
        bit_tmp += 1
        prob = prob / prob_sum
        groups = grouping(prob)
    return return_bits

# 简单解码器，不应用任何编码
def PLAIN_decoder(prob, prev, **kwargs):
    return ""

# 解码器主要函数，根据指定算法选择合适的解码函数
def decoder(alg, prob, prev, cur_interval=None, **kwargs):
    if alg.lower() == "plain":
        return PLAIN_decoder(prob, **kwargs)
    if alg.lower() == "flc":
        return FLC_decoder(prob, prev, **kwargs)
    if alg.lower() == "hc":
        return HC_decoder(prob, prev, **kwargs)
    if alg.lower() == "ac":
        return AC_decoder(prob, prev, cur_interval, **kwargs)
    if alg.lower() == "adg":
        return ADG_decoder(prob, prev, **kwargs)
    if alg.lower() == "adgv2":
        return ADG_V2_decoder(prob, prev, **kwargs)
    raise ValueError("no such algorithm")

# 用于从配置文件中提取解码器的配置参数
def decoder_configs(configs:ConfigParser, alg):
    kwargs = {}
    for k in configs.options("generate"):
        if k not in ["bit_filepath", "model_name", "max_length", "generate_num", "alg"]:
            kwargs[k] = configs.get("generate", k, )
    if alg.lower() == "plain":
        new_kwargs = {}
        for k in ["topp",]:
            if  kwargs.get("topp") is not None:
                new_kwargs[k] = configs.getfloat("generate", "topp")
        for k in ["topk",]:
            if  kwargs.get("topk") is not None:
                new_kwargs[k] = configs.getint("generate", "topk")
        for k in ["do_sample", ]:
            if kwargs.get("do_sample") is not None:
                new_kwargs[k] = configs.getboolean("generate", "do_sample")
        return new_kwargs
    if alg.lower() == "flc":
        new_kwargs = {}
        for k in ["bit", ]:
            if kwargs.get("bit") is not None:
                new_kwargs[k] = configs.getint("generate", "bit")
        return new_kwargs
    if alg.lower() == "hc":
        new_kwargs = {}
        for k in ["bit", ]:
            if kwargs.get("bit") is not None:
                new_kwargs[k] = configs.getint("generate", "bit")
        return new_kwargs
    if alg.lower() == "ac":
        new_kwargs = {}
        for k in ["precision", ]:
            if kwargs.get("precision") is not None:
                new_kwargs[k] = configs.getint("generate", "precision")
        return new_kwargs
    if alg.lower() == "adg":
        return {}
    if alg.lower() == "adgv2":
        new_kwargs = {}
        for k in ["epsilon", ]:
            if kwargs.get("epsilon") is not None:
                new_kwargs[k] = configs.getfloat("generate", "epsilon")
        for k in ["max_bit", ]:
            if kwargs.get("max_bit") is not None:
                new_kwargs[k] = configs.getint("generate", "max_bit")
        return new_kwargs
    raise ValueError("no such algorithm")
