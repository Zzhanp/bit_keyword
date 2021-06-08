#-*- codeing= utf-8 -*-
#@Time : 2021/5/26 20:03
from pyhanlp import *
import numpy as np
import gc
import csv
import xlwt
import help_func as func
import pandas as pd
from collections import Counter
from PIL import Image
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from wordcloud import WordCloud  # 词云图相关


def del_data():
    csv_path = os.getcwd() + '\\' + 'output.csv'
    xlsx_path = os.getcwd() + '\\' + 'output.xlsx'
    if os.path.exists(csv_path):
        os.remove(csv_path)
    elif os.path.exists(xlsx_path):
        os.remove(xlsx_path)
        print('历史记录清除完毕')
    else:
        print('历史记录已经是空的啦') 
    del csv_path
    del xlsx_path
    gc.collect()

def csv_to_xlsx():
    csv = pd.read_csv('output.csv', header=None, usecols=[0,1],index_col=None)
    csv.to_excel('output.xlsx', sheet_name='data',header=None,index=None)

def read_csv():
    csv_path = os.getcwd() + '\\' + 'output.csv'
    if not os.path.exists(csv_path):
        print('历史记录是空的')
        del csv_path
        gc.collect()
        return 0
    else:
        df = pd.read_csv(csv_path, header=None, usecols=[0,1])
        dict1 = df.to_dict()
        dict_word =dict1[1]
        dict_word = [i for i in dict_word.values()]
        dict_title = dict1[0]
        dict_title = [i for i in dict_title.values()]
        #print(dict_word,dict_title)
        d = []
        for i in range(len(dict_title)):
            s = dict_title[i]
            k = dict_word[i]
            dict_f = dict([(s,k)])
            d.append(dict_f)
        del df
        del dict1
        del dict_word
        del dict_title
        del csv_path
        gc.collect()
        return d


def draw_word_cloud(words_dict, title, savepath='./word_cloud'):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    wc = WordCloud(font_path='simhei.ttf', background_color='white', max_words=2000, width=1920, height=1080, margin=5)
    wc.generate_from_frequencies(words_dict)
    wc.to_file(os.path.join(savepath, title + '.png'))
    #wc.to_file('{}{}.png'.format(savepath,"/111"))
    #print(os.path.join(savepath, title+'.png'))      绝对路径和相对路径！！！
    #wc.to_file('./word_cloud/{}.png'.format(title))
    path_c = os.getcwd() + '\\word_cloud\\' + title + '.png'
    print('词云图保存在{}'.format(path_c))
    del wc
    del path_c
    gc.collect()


def save(save_filename, save_contents):
    save_file = open(save_filename, 'w+', encoding='utf-8')
    save_file.write(save_contents)
    save_file.close()
    print('pdf转txt成功，保存在{}'.format(save_filename))


def extract_pdf_content(pdf):
    rsrcmgr = PDFResourceManager()
    codec = 'utf-8'
    outfp = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr=rsrcmgr, outfp=outfp,  laparams=laparams)
    with open(pdf, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)
    mystr = outfp.getvalue()
    device.close()
    outfp.close()
    return mystr


def get_single_pdf(sin_pdf_path):
    sin_pdf_dict = {}
    sin_pdf_key = sin_pdf_path.split("\\")[-1]
    sin_pdf_dict[sin_pdf_key] = extract_pdf_content(sin_pdf_path)
    return sin_pdf_dict


def extract(file, key_num):
    # 获取文件名
    txt_name = file.split("\\")[-1]

    file_de = file.split('.')[-1]

    if file_de == 'pdf':
        path_txt = os.getcwd() + '\\' + txt_name[:-3] + 'txt'
        my_dict = get_single_pdf(file)
        # 保存到txt
        save(path_txt, my_dict[txt_name])
        filename = path_txt

    filename = file

    NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    print('正在处理{}'.format(txt_name))
    # 设置关键词词性
    pos = ['nn', 'n', 'g', 'nnd', 'gb', 'nb', 'nnt', 'gbc', 'nba', 'gc', 'nbc', 'gg', 'nbp',
           'gi', 'nf', 'gm', 'gp', 'nh', 'ns', 'nhd', 'nsf', 'nhm', 'nt', 'ni', 'ntc', 'nic', 'ntcb'
        , 'ntcf', 'nit',
           'ntch', 'nth', 'nts', 'nm', 'nto', 'ntu', 'nmc']

    # 创建一个停用词列表
    stopwords1 = func.stopwordslist()

    # 打开文件
    print('开始分词')
    with open(filename, encoding='utf-8', errors='ignore') as f:
        # 读取文本
        k = f.read()
        # 将换行删除掉
        k = k.replace('\n', '')
        k = k.replace(' ', '')
        # 对文本进行分词
        title_key = []
        text_segment = NLPTokenizer.segment(k)
        print('hanlp分词完毕')
        # 提取标题关键词
        text_ab = HanLP.extractSummary(k, 3)
        title_word = HanLP.extractKeyword(text_ab[0] + text_ab[1]+text_ab[2], 5)
        for title_word1 in title_word:
            if title_word1 not in stopwords1 and len(title_word1) > 1:
                title_key.append(title_word1)
        # 文本词语列表
        word_list = []
        # 筛选符合词性的词语
        for word_seg in text_segment:
            if str(word_seg.nature) in pos and len(word_seg.word) > 1:
                word_list.append(word_seg.word)

    # 输出结果为outstr
        outstr = ''
        print('分词成功，开始去除停用词')
    # 去停用词
        for word in word_list:
            if word not in stopwords1:
                if word != '\t':
                    outstr += word
                    outstr += " "

        outstr1 = outstr.split(' ')

    # 去除空字符
        outstr1 = list(filter(lambda x: x != '', outstr1))

        word1 = []

    # 去除纯字母加数字
        for word in outstr1:
            delete = all(ord(c) < 128 for c in word)
            if delete == 0:
                word1.append(word)
    # 计算标题相似度
        text_word = []
        # text_word2 是单词列表
        text_word2 = dict(Counter(word1))
        for key in text_word2:
            text_word.append(key)
        title_sim = np.zeros((len(title_key), len(text_word)))
        title_sim = func.title_similarity(title_sim, title_key, text_word)
        title_sim_1 = np.zeros((1, len(text_word)))
        for i in range(len(title_key)):
            title_sim_1 = np.add(title_sim[i],title_sim_1)
        title_sim_1 = title_sim_1/np.max(title_sim_1)

    # 求关键词的分数矩阵
        # word3 是dict，单词对应得分
        print('开始建立自信矩阵')
        word3 = func.confidence(word1, title_sim_1)
        print('自信矩阵建立成功')
        print('开始建立相似度矩阵')
    # 创建相似度矩阵
        w_sim1 = np.zeros((len(text_word), len(text_word)))

    # 计算每个候选词之间的词义相似度
        w_sim1 = func.similarity(w_sim1, text_word)
        print('开始进行博弈')
    # 所有候选关键词进行博弈
        rank_num = 50
        score = func.new_rank(word3, text_word, w_sim1, rank_num)
        print('博弈结束，准备输出')
    # 把最终得分和候选词对应
        rank = dict(zip(text_word, score))

    # 判断输入是否太多
    if len(rank) < int(key_num):
        print("要求显示的关键词超出了提取的关键词，只显示提取到的所有关键词")
    # 对候选词得分进行排序，选出分数前十
    rank1 = func.order_dict(rank, len(rank))
    print(rank1)
    keyword = []
    for i, key in enumerate(rank1):
        if i < int(key_num):
            keyword.append(key)
        else:
            break
    print(keyword)
    func.to_csv(keyword, filename)
    csv_to_xlsx()
    print('生成词云中')
    draw_word_cloud(text_word2, txt_name.split('.')[0])
    print('词云生成完毕')
    del txt_name
    del rank1
    del rank
    del score
    del rank_num
    del word3
    del word_list
    del text_word
    del outstr1
    del outstr
    del text_segment
    del pos
    del stopwords1
    del NLPTokenizer
    del text_word2
    gc.collect()
    return keyword


'''file_nam = r'E:\my code\python\keyword9.0\keyword1.0\人工智能与机器人.txt'    #用绝对路径，相对路径会报错
key_num1 = 20
keyword1 = extract(file_nam, key_num1)'''

#增加csv_to_xlsx
#修改del_csv：增加删除xlsx