import numpy as np
from scipy import stats


def select_top(data_list, value):#valueには上位〇%の〇を入れる
  sums = np.array([np.sum(arr) for arr in data_list])
  print("平均値: ", np.mean(sums))
  #閾値計算
  parcent = 100 - value
  threshold = np.nanpercentile(sums, parcent)
  print("閾値: ", threshold)
  #上位〇%を抽出
  top_quarter_arrays = [arr for arr, s in zip(data_list, sums) if s >= threshold]
  
  return top_quarter_arrays


#二つのデータセットからK-S検定で異なると判断された変数の番号とP値が格納された辞書を返す。
def KS_test(data_01, data_02):
  data_num = len(data_01[0])
  num_different_parameters = 0
  different_latent_dic = {}
  
  for i in range(data_num): 
      ks_statistic, p_value = stats.ks_2samp(data_01[:, i], data_02[:, i])
      alpha = 0.01  # 有意水準
      if p_value < alpha:
          num_different_parameters += 1
          different_latent_dic[i] = p_value
          # print(f"Parameter {i} is different (p-value: {p_value:.4f})")
  
  # 異なる分布のパラメータの個数を表示
  print(f"Number of parameters with different distributions: {num_different_parameters}")

  return different_latent_dic


def sort_dict(dic, reverse=False, print=False):
  sorted_items = dict(sorted(dic.items(), key=lambda x: x[1], reverse=reverse))

  if print == True:
    for key, value in sorted_items.items():
        print(f"{key}: {value},")

  return sorted_items
  
