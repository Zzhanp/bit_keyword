#-*- codeing= utf-8 -*-
#@Time : 2021/5/26 20:03
from collections import Counter
import numpy as np
import pandas as pd
import os
import gc
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
import numba as nb
cred = credential.Credential("AKIDOcbJkuoRmqd8NL2RZR2MPBgurXw4uUsz", "sUor9BwQcYDbiX6NO7ZkeIfndCVXJq7C")
httpProfile = HttpProfile()
httpProfile.endpoint = "nlp.tencentcloudapi.com"
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = nlp_client.NlpClient(cred, "ap-guangzhou", clientProfile)
req = models.TextSimilarityRequest()


# 将列表分组函数，每99个分一组
def div_list(listTemp, n):
    listr = []
    for i in range(0, len(listTemp), n):
        listr.append(listTemp[i:i + n])
    return listr


# 装换上三角矩阵的函数
@nb.njit()
def sym(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[j, i] = a[i, j]
    return a


# 进行博弈论的函数
def new_rank(new_rank_dict, new_rank_word_list, new_rank_sim, new_interact):
    # 创建位置频率矩阵
    position_freq_scores = []
    for new_rank_key in new_rank_dict:
        position_freq_scores.append(new_rank_dict[new_rank_key])

    # 转换成np数组
    con = position_freq_scores
    con = np.array(con)
    con = con.astype(np.float)

    del position_freq_scores

    # 创建博弈矩阵
    game = np.zeros(shape=(2, 2))
    # n次的得分矩阵
    stream_demo = [[0.0 for y in range(new_interact)] for x in range(len(new_rank_word_list))] # wordlist 行 * intera 列

    # --  GAME TEHORETIC ALGO  --#
    prob = []
    temp = np.zeros(shape=(2, 1))

    # 初始化权重矩阵
    for y in range(len(new_rank_word_list)):
        prob.append([[0.5], [0.5]])

    prob_rep = np.zeros(shape=(len(new_rank_word_list), 2))

    for i in range(new_interact):
        # print("*********第{}轮博弈**********".format(i))
        # print(prob)
        for index1, word1 in enumerate(new_rank_word_list):

            u1 = np.zeros(shape=(2, 1))
            u2 = 0.0
            for index2, word2 in enumerate(new_rank_word_list):
                # print("*****{} 正在与 {}博弈 *****".format(word1,word2))

                if word1 != word2:
                    # KG PAYOFFS
                    # ij都是关键词
                    game[0, 0] = con[index2] * new_rank_sim[index1][index2] + con[index1] * (
                                1 - con[index2])  # Cj *S + (1-Cj) * Ci
                    # i是关键词， j不是关键词
                    game[0, 1] = con[index1] * con[index2] + (1 - new_rank_sim[index1][index2]) * (
                                1 - con[index2])  # Cj * Ci +(1-Cj) * (1- S)
                    # i不是关键词 j 是关键词
                    game[1, 0] = (1 - new_rank_sim[index1][index2]) * con[index2] + (1 - con[index2]) * (
                                1 - con[index1])  # Cj * (1-S) + (1 - Cj) * (1-Ci)
                    # ij都不是关键词
                    game[1, 1] = (1 - con[index1]) * con[index2] + (new_rank_sim[index1][index2]) * (
                                1 - con[index2])  # Cj * (1-Ci) + (1 - Cj) * S

                    temp = np.dot(game, np.asarray(prob[index2]))  # 2*2 与 2*1矩阵相乘
                    u1 = np.add(u1, temp)  # 2*1矩阵
                    u2 = u2 + np.dot(np.transpose(prob[index1]), temp)[0][0]

            prob_rep[index1] = np.transpose(np.multiply(prob[index1], u1 / u2))
            del u1
            del u2

        prob = []
        # 将新的概率矩阵写入prob
        for y in range(len(new_rank_word_list)):
            prob.append([[prob_rep.tolist()[y][0]], [prob_rep.tolist()[y][1]]])
        # 写入每次博弈成为关键词的概率
        for j in range(len(new_rank_word_list)):
            stream_demo[j][i] = prob_rep[j, 0]

    new_rank_score = []

    for k in range(len(new_rank_word_list)):
        new_rank_score.append(sum(stream_demo[k]))

    del stream_demo
    del prob
    del prob_rep
    del temp
    del game
    del con
    gc.collect()
    # 返回权重矩阵
    return new_rank_score


def to_csv(word_k, path_f):
    dict1 = {}
    path_t = os.getcwd() + '\\output.csv'
    dict1[path_f] = ','.join(word_k)
    df = pd.DataFrame.from_dict(dict1, orient='index').reset_index()
    df.columns = ["path", "content"]
    df.to_csv(path_t, mode='a', index=False,header=False,encoding="utf-8")
    del df
    del dict1
    del path_t
    gc.collect()


# word_list1 所有关键词
# title_sim 标题相似度矩阵
# 返回一个字典
def confidence(word_list1, title_sim):
    wordkey = []
    wordgui = []
    num = len(word_list1)
    word2 = dict(Counter(word_list1))
    for index, wordt in enumerate(word2):
        scores = 0
        for j in range(len(word_list1)):
            if wordt == word_list1[j]:
                scores += 1/(1000+j)
        wordgui.append(scores)

    max_word = max(wordgui)
    # min_word = min(wordgui)
    # word_gui_final = [(x - min_word) / (max_word - min_word) for x in wordgui]
    word_gui_final = [x / max_word for x in wordgui]
    for i in range(len(word_gui_final)):
        word_gui_final[i] = word_gui_final[i] + title_sim[0][i]
    max_word = max(word_gui_final)
    min_word = min(word_gui_final)
    word_gui_final = [(x - min_word) / (max_word - min_word) for x in word_gui_final]
    # word_gui_final = [x / max_word for x in wordgui]
    del max_word
    # del min_word
    for key in word2:
        wordkey.append(key)
    final = dict(zip(wordkey, word_gui_final))
    del wordgui
    del num
    del word2
    del word_gui_final
    del wordkey
    gc.collect()
    return final


def stopwordslist():
    stopwords = [line.strip() for line in open('stopword_china.txt', encoding='UTF-8').readlines()]
    return stopwords


def order_dict(dicts, n):
    result = []
    result1 = []
    result2 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
        result2.append(r[1])
    rankr = dict(zip(result1, result2))
    return rankr


# 用腾讯AI进行相似度比较
# 上三角矩阵，只用求一半
def similarity(text_sim, text_word,):
    global req
    global client
    text_len = len(text_word)
    print('共{}个候选词'.format(text_len))
    for i, text_word1 in enumerate(text_word):
        if i % 5 == 0:
            print('还剩{}个'.format(text_len-i))
        text_remain = i-text_len
        word_remain_word = text_word[text_remain:]
        if len(word_remain_word) < 100:
            try:
                params = {
                    "SrcText": text_word1,
                    "TargetText": word_remain_word
                }
                req.from_json_string(json.dumps(params))

                resp = client.TextSimilarity(req)
                text_sim1 = resp.to_json_string()
                text_json = json.loads(text_sim1)
                text_sim2 = text_json['Similarity']
                # 返回的是一个倒序排列的json，得把每个的相似度提取出来
                for j, text_word2 in enumerate(word_remain_word):
                    for index, text_word3 in enumerate(text_sim2):
                        if text_word3['Text'] == text_word2:
                            text_sim[i][i + j] = text_word3['Score']
                            break
                del resp
                del text_sim1
                del text_json
                del text_sim2
                del params

            except TencentCloudSDKException as err:
                print(err)

        else:
            text_indexy = 0
            for text_list1 in div_list(word_remain_word, 99):
                try:
                    params = {
                        "SrcText": text_word1,
                        "TargetText": text_list1
                    }
                    req.from_json_string(json.dumps(params))

                    resp = client.TextSimilarity(req)
                    text_sim1 = resp.to_json_string()
                    jss = json.loads(text_sim1)
                    text_sim2 = jss['Similarity']
                    # 返回的是一个倒序排列的json，得把每个的相似度提取出来
                    for j, text_word2 in enumerate(text_list1):
                        for index, text_word3 in enumerate(text_sim2):
                            if text_word3['Text'] == text_word2:
                                text_sim[i][i + text_indexy] = text_word3['Score']
                                text_indexy = text_indexy+1
                                break
                    del resp
                    del params
                    del text_sim1
                    del jss
                    del text_sim2

                except TencentCloudSDKException as err:
                    print(err)
        del text_remain
        del word_remain_word
    print('相似度上三角矩阵计算完毕')
    text_sim2 = text_sim.astype(np.float)
    text_final = sym(text_sim2)
    print('相似度矩阵计算完毕，将进行归一化')
    # 将数组归一化
    # y = (text_final - np.min(text_final)) / (np.max(text_final) - np.min(text_final))
    y = text_final / np.max(text_final)
    del text_final
    del text_sim2
    del text_len
    gc.collect()
    return y


# 标题相似度矩阵
def title_similarity(title_sim_array, title_word, text_word):
    global req
    global client
    for i, title_word1 in enumerate(title_word):
        if len(text_word) < 100:
            try:
                params = {
                    "SrcText": title_word1,
                    "TargetText": text_word
                }
                req.from_json_string(json.dumps(params))

                resp = client.TextSimilarity(req)
                text_sim1 = resp.to_json_string()
                text_json = json.loads(text_sim1)
                text_sim2 = text_json['Similarity']
                # 返回的是一个倒序排列的json，得把每个的相似度提取出来
                for j, text_word2 in enumerate(text_word):
                    for index, text_word3 in enumerate(text_sim2):
                        if text_word3['Text'] == text_word2:
                            title_sim_array[i][j] = text_word3['Score']
                            break
                del resp
                del text_sim1
                del text_json
                del text_sim2
                del params

            except TencentCloudSDKException as err:
                print(err)

        else:
            text_indexy = 0
            for text_list1 in div_list(text_word, 99):
                try:
                    params = {
                        "SrcText": title_word1,
                        "TargetText": text_list1
                    }
                    req.from_json_string(json.dumps(params))

                    resp = client.TextSimilarity(req)
                    text_sim1 = resp.to_json_string()
                    jss = json.loads(text_sim1)
                    text_sim2 = jss['Similarity']
                    # 返回的是一个倒序排列的json，得把每个的相似度提取出来
                    for j, text_word2 in enumerate(text_list1):
                        for index, text_word3 in enumerate(text_sim2):
                            if text_word3['Text'] == text_word2:
                                title_sim_array[i][text_indexy] = text_word3['Score']
                                text_indexy = text_indexy+1
                                break
                    del resp
                    del params
                    del text_sim1
                    del jss
                    del text_sim2

                except TencentCloudSDKException as err:
                    print(err)

    text_final = title_sim_array.astype(np.float)

    # 将数组归一化
    # y = (text_final - np.min(text_final)) / (np.max(text_final) - np.min(text_final))
    y = text_final
    del text_final
    gc.collect()
    return y